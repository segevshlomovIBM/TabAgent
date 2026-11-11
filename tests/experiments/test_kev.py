"""The Kev baseline (experiments/baselines/kev.py) against a fake Kev server on localhost.

No network, no GPU and no model: the fake server answers from the question text, so these tests check the
client's own logic (what is sent, how answers map back, retries, the cache, the per-fold choice of wording),
not Kev's behaviour. The tables are fabricated.
"""

from __future__ import annotations

import http.server
import json
import re
import threading

import numpy as np
import pandas as pd
import pytest

from experiments.baselines import kev as K
from tabagent.table import DecisionSpec, DecisionTable

TOOLS = ["search_flights", "book_flight", "cancel_booking", "get_weather", "send_email", "update_address"]
SECRET = "the-gold-answer"


def tau_table(n_tasks: int = 10, runs: int = 2) -> DecisionTable:
    rows = [
        {
            "task_id": f"t{t}",
            "traj_uid": f"t{t}-r{r}",
            "instruction": f"please help with case {t}",
            "tool_name": tool,
            "tool_desc": SECRET,  # the paper's baselines read the name only on τ², so this must not be sent
            "used": int(j == t % len(TOOLS)),
            "reward": 1.0,
            "score": SECRET,
        }
        for t in range(n_tasks)
        for r in range(runs)
        for j, tool in enumerate(TOOLS)
    ]
    spec = DecisionSpec(
        "ranking", group="task_id", label="used", candidate="tool_name", decision=("traj_uid",)
    )
    return DecisionTable(pd.DataFrame(rows), spec)


FOLDS = {f"t{t}": t % 5 for t in range(10)}


def tool_score(text: str) -> float:
    """The fake server's answer: a fixed score per tool, read from the question text."""
    return (TOOLS.index(re.search(r"Tool: (\w+)", text).group(1)) + 1) / 10


class Fake(http.server.BaseHTTPRequestHandler):
    bodies: list = []
    fail_first, too_long, down = 0, 10**9, set()

    def log_message(self, *a):
        pass

    def do_POST(self):
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        cls.bodies.append(body)
        if next(iter(body["state"].values())) in cls.down:
            return self.send_error(500)
        if cls.fail_first > 0:
            cls.fail_first -= 1
            return self.send_error(503)
        if len(json.dumps(body["state"])) > cls.too_long:
            return self.send_error(422)
        ans = {}
        for k, q in body["questions"].items():
            if q["type"] == "noul":
                ans[k] = {"type": "noul", "noul": tool_score(q["instructions"])}
            else:
                uniform = {c: 1 / len(q["criteria"]) for c in q["criteria"]}
                ans[k] = {"type": "choice", "probabilities": uniform}
        out = json.dumps({"answers": ans, "latency_ms": 1.0}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def server():
    Fake.bodies, Fake.fail_first, Fake.too_long, Fake.down = [], 0, 10**9, set()
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def score(table, url, cache, dataset="tau2/airline", **client):
    c = K.Client(url, backoff=0, **client)
    return K.kev_scores(table, dataset, FOLDS, ("p@r",), url=url, revision="r", cache=cache, client=c)


def test_every_wording_asks_once_per_decision_and_a_rerun_asks_nothing(tmp_path, server):
    t = tau_table()
    _, config = score(t, server, tmp_path / "c.jsonl")
    n = t.decision_ids.nunique()
    assert len(Fake.bodies) == n * len(K.VARIANTS) and config["from_cache"] == 0
    # one choice question per listwise request
    assert sum(len(b["questions"]) == 1 for b in Fake.bodies) == n * 3
    _, again = score(t, server, tmp_path / "c.jsonl")
    assert len(Fake.bodies) == n * len(K.VARIANTS) and again["from_cache"] == n * len(K.VARIANTS)


def test_no_label_outcome_or_unread_column_reaches_kev(tmp_path, server):
    score(tau_table(), server, tmp_path / "c.jsonl")
    sent = json.dumps(Fake.bodies)
    assert SECRET not in sent
    assert all(set(b["state"]) == {"instruction"} for b in Fake.bodies)


def test_pointwise_scores_map_back_to_the_candidate_they_asked_about(tmp_path, server):
    t = tau_table()
    _, config = score(t, server, tmp_path / "c.jsonl")
    want = t.df["tool_name"].map(lambda x: (TOOLS.index(x) + 1) / 10).to_numpy()
    reqs = K.requests(t, "tau2/airline")
    got, _ = K.answers(reqs, "tau2", K.Client(server), "r", tmp_path / "c.jsonl")
    ids = t.decision_ids.astype(str).to_numpy()
    p1 = np.array([got[("P1", d)]["scores"][c] for d, c in zip(ids, t.df["tool_name"], strict=True)])
    assert np.allclose(p1, want) and config["requests"] == len(reqs) * len(K.VARIANTS)


def test_candidate_order_depends_on_the_seed_and_the_decision_only():
    d = K.requests(tau_table(1, 1), "tau2/airline")[0]
    flipped = {**d, "candidates": list(reversed(d["candidates"]))}
    order = [x["id"] for x in K.shuffled(d, 1)]
    assert order == [x["id"] for x in K.shuffled(flipped, 1)] and order != [x["id"] for x in K.shuffled(d, 2)]


def test_no_fold_picks_its_wording_from_its_own_rows():
    t = tau_table()
    y = t.df["used"].to_numpy().astype(float)
    fold0 = t.df["task_id"].map(FOLDS).to_numpy() == 0
    vec = {v: np.full(len(y), 0.5) for v in K.VARIANTS}
    vec["L3"] = np.where(fold0, y, 1 - y)  # perfect on fold 0 alone, worst everywhere else
    _, chosen = K.select(t, FOLDS, vec)
    assert chosen[0] == "P1"  # fold 0's own rows would have picked L3
    assert all(chosen[f] == "L3" for f in (1, 2, 3, 4))


def test_a_failure_is_retried_and_a_request_that_keeps_failing_stops_the_run(tmp_path, server):
    Fake.fail_first = 2
    _, config = score(tau_table(2, 1), server, tmp_path / "a.jsonl")
    assert config["errors"] == {"http503": 2}
    Fake.down = {"please help with case 1"}
    with pytest.raises(SystemExit, match="unscored"):
        score(tau_table(2, 1), server, tmp_path / "b.jsonl", retries=1)


def test_a_state_too_long_for_kev_is_cut_in_the_middle_and_counted(tmp_path, server):
    t = tau_table(1, 1)
    t = t.with_df(t.df.assign(instruction="x" * 400))
    Fake.too_long = 300
    _, config = score(t, server, tmp_path / "c.jsonl")
    assert config["cut_states"] == len(K.VARIANTS) and "[...]" in Fake.bodies[-1]["state"]["instruction"]


def test_the_wordings_are_the_ones_the_published_numbers_used():
    d = {"decision": "x", "app": "airline", "state": {"instruction": "i"}}
    d["candidates"] = [{"id": "book", "desc": ""}]
    q, ids = K.build(d, "P2", "tau2")
    assert q["c0"]["instructions"] == (
        "An AI agent is handling the customer request in this state and is about to pick from its airline "
        "tools. Tool: book"
    )
    assert q["c0"]["criteria"] == {
        "true": "This tool is needed to complete the request.",
        "false": "This tool is not needed to complete the request.",
    }
    assert ids == {"c0": "book"} and K.build(d, "L1", "tau2")[0]["list"]["criteria"] == {"book": ""}
    s = {"decision": "y", "app": "", "state": {"task_text": "t"}}
    s["candidates"] = [{"id": "s", "desc": "long text", "list_desc": "long"}]
    assert K.build(s, "L3", "skill")[0]["list"]["instructions"].endswith(
        "Which skill belongs at the top of the shortlist for the task?"
    )
    assert K.build(s, "L3", "skill")[0]["list"]["criteria"] == {"s": "long"}
    assert K.build(s, "P1", "skill")[0]["c0"]["instructions"].endswith("Skill: s: long text")


def test_the_order_of_options_is_part_of_the_cache_key():
    a, b = {"list": {"criteria": {"x": "1", "y": "2"}}}, {"list": {"criteria": {"y": "2", "x": "1"}}}
    assert K.key_of("r", {"s": 1}, a) != K.key_of("r", {"s": 1}, b)


def test_a_table_without_wordings_is_refused():
    with pytest.raises(SystemExit, match="kev-zs is set up for"):
        K.requests(tau_table(1, 1), "mind2web/operation")


def test_the_harness_writes_kev_like_any_baseline(tmp_path, server, monkeypatch):
    pytest.importorskip("torch")  # the harness imports the DSR and LLM baselines
    from experiments.baselines import run as R

    t = tau_table()
    monkeypatch.setattr(R.datasets, "load", lambda name, **kw: t)
    monkeypatch.setattr(R.datasets, "folds", lambda name: FOLDS)
    out = tmp_path / "kev.json"
    args = ["--dataset", "tau2/airline", "--method", "kev-zs", "--model", "jaredpalmer/kev-27b"]
    R.main([*args, "--kev-url", server, "--kev-cache", str(tmp_path / "c.jsonl"), "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["seeds"] == [None] and res["model"] == "jaredpalmer/kev-27b"
    assert len(res["per_decision"]["p@r"]) == t.decision_ids.nunique()
    assert res["config"]["revision"] == "jaredpalmer/kev-27b"
    assert set(map(int, res["config"]["chosen_by_fold"])) == set(range(5))
