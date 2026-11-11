"""Kev as one more baseline: a served decision model, scored zero-shot through the paper's harness.

Kev (https://github.com/jaredpalmer/kev) answers typed questions about a state. Each decision becomes one
label-free request per question wording, built from the preset's text view: the state is the customer's
instruction on τ² and the task on skill selection; a candidate is the tool name, or the skill name and
description. Three wordings ask about each candidate (Kev's ``noul`` question) and three ask for one choice
over all of them (``choice``). For each test fold, the wording with the best mean P@R on the other folds
supplies the scores, so no choice sees its own fold. Kev runs in its own server; this module only talks to it
over HTTP::

    git clone https://github.com/jaredpalmer/kev && cd kev && uv sync --extra serve
    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b --port 8008
    python -m experiments.baselines.run --dataset tau2/airline --method kev-zs --model jaredpalmer/kev-27b

Every answer is appended to a JSON-lines cache keyed on a hash of the revision, the state and the questions,
so a rerun asks only for what is missing. ``KEV_API_KEY`` is sent as a bearer key when it is set.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent.evaluation import metrics as M
from tabagent.table import DecisionTable

from .common import fold_array

VARIANTS = ("P1", "P2", "P3", "L1", "L2", "L3")
POINTWISE = ("P1", "P2", "P3")
SHUFFLE_SEED = 20260720
# a listwise question cuts each skill description to this, so its 195 options stay within 8,192 tokens
LIST_CHARS = 110

WORDINGS = {
    "tau2": {
        "instr": (
            "An AI agent is handling the customer request in this state and is about to pick from its "
            "{app} tools."
        ),
        "noun": "tool",
        "verb": "call",
        "work": "request",
    },
    "skill": {
        "instr": "An AI agent has the task in this state and is about to pick from its skills.",
        "noun": "skill",
        "verb": "use",
        "work": "task",
    },
}
# dataset -> (wording, its {app} word, state column, candidate id column, candidate description column)
TABLES = {
    "tau2/airline": ("tau2", "airline", "instruction", "tool_name", None),
    "tau2/retail": ("tau2", "retail", "instruction", "tool_name", None),
    "tau2/telecom": ("tau2", "telecom", "instruction", "tool_name", None),
    "tau2/all-domains": ("tau2", "available", "instruction", "tool_name", None),
    "skillsbench/skill-select": ("skill", "", "task_text", "skill_name", "skill_desc"),
}


def requests(table: DecisionTable, dataset: str) -> list[dict]:
    """One request per decision, from the state column and each candidate's id and description only."""
    if dataset not in TABLES:
        raise SystemExit(f"kev-zs is set up for {', '.join(TABLES)}, not {dataset}")
    _, app, state_col, id_col, desc_col = TABLES[dataset]
    df = table.df.assign(_d=table.decision_ids.astype(str).to_numpy())
    out = []
    for d, g in df.groupby("_d", sort=False):
        descs = g[desc_col].astype(str) if desc_col else pd.Series("", index=g.index)
        cands = [{"id": c, "desc": x} for c, x in zip(g[id_col].astype(str), descs, strict=True)]
        if desc_col:
            for c in cands:
                c["list_desc"] = c["desc"][:LIST_CHARS].rstrip()
        if len({c["id"] for c in cands}) != len(cands):
            raise ValueError(f"{dataset}: duplicate candidate ids in decision {d}")
        state = {state_col: str(g[state_col].iloc[0])}
        out.append({"decision": d, "app": app, "state": state, "candidates": cands})
    return out


def shuffled(d: dict, seed: int = SHUFFLE_SEED) -> list[dict]:
    """The candidates in an order that depends only on the seed and the decision id, never on a label."""
    c = sorted(d["candidates"], key=lambda x: x["id"])
    random.Random(hashlib.sha256(f"{seed}|{d['decision']}".encode()).hexdigest()).shuffle(c)
    return c


def build(d: dict, variant: str, wording: str, seed: int = SHUFFLE_SEED) -> tuple[dict, dict | None]:
    """The questions of one request, and for the pointwise wordings the candidate behind each question id."""
    w = WORDINGS[wording]
    n, v, k = w["noun"], w["verb"], w["work"]
    instr = w["instr"].format(app=d.get("app", ""))
    c = shuffled(d, seed)
    if variant in POINTWISE:
        yes, no = {
            "P1": (
                f"The agent will {v} this {n} for the {k}.",
                f"The agent will not {v} this {n} for the {k}.",
            ),
            "P2": (
                f"This {n} is needed to complete the {k}.",
                f"This {n} is not needed to complete the {k}.",
            ),
            "P3": (
                f"This {n} belongs on the shortlist for the {k}.",
                f"This {n} does not belong on the shortlist for the {k}.",
            ),
        }[variant]
        # a name-only view has no description, and no colon after the name
        text = [f"{x['id']}: {x['desc']}" if x["desc"] else x["id"] for x in c]
        q = {
            f"c{i}": {
                "type": "noul",
                "instructions": f"{instr} {n.capitalize()}: {t}",
                "criteria": {"true": yes, "false": no},
            }
            for i, t in enumerate(text)
        }
        return q, {f"c{i}": x["id"] for i, x in enumerate(c)}
    ask = {
        "L1": f"Which {n} will the agent {v} for the {k}?",
        "L2": f"Which {n} is most needed to complete the {k}?",
        "L3": f"Which {n} belongs at the top of the shortlist for the {k}?",
    }[variant]
    criteria = {x["id"]: x.get("list_desc", x["desc"]) for x in c}
    return {"list": {"type": "choice", "instructions": f"{instr} {ask}", "criteria": criteria}}, None


def key_of(revision: str, state: dict, questions: dict) -> str:
    """Hash of revision, state and questions. Keys are not sorted: the order of a question's options is part
    of the question."""
    return hashlib.sha256(json.dumps([revision, state, questions], ensure_ascii=False).encode()).hexdigest()


class Client:
    """Posts one request to a Kev server, retrying failures; a 422 (state too long) cuts the middle of the
    state, at most twice, and the cut is recorded."""

    def __init__(self, url: str, timeout: float = 900, retries: int = 4, backoff: float = 20.0):
        self.url, self.timeout, self.retries, self.backoff = url.rstrip("/"), timeout, retries, backoff
        self.head = {"Content-Type": "application/json"}
        if os.environ.get("KEV_API_KEY"):
            self.head["Authorization"] = "Bearer " + os.environ["KEV_API_KEY"]
        self.errors: Counter = Counter()

    def post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url + "/v1/systemone", data=data, headers=self.head)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def shrink(state: dict) -> dict:
        """The state with the middle of its longest text field cut to half that field's length."""
        field = max((k for k, v in state.items() if isinstance(v, str)), key=lambda k: len(state[k]))
        t, q = state[field], len(state[field]) // 4
        return {**state, field: f"{t[:q]} [...] {t[-q:]}"}

    def ask(self, d: dict, variant: str, wording: str) -> dict | None:
        """One request's scores by candidate id, with its cuts and timing; None when every attempt failed."""
        state, cuts, attempt = d["state"], 0, 0
        qs, ids = build(d, variant, wording)
        while attempt < self.retries:
            t0 = time.perf_counter()
            try:
                r = self.post({"model": "kev-latest", "state": state, "questions": qs})
            except urllib.error.HTTPError as e:
                self.errors[f"http{e.code}"] += 1
                if e.code == 422 and cuts < 2:
                    state, cuts = self.shrink(state), cuts + 1
                    continue
                attempt += 1
                if e.code < 500:
                    return None
                time.sleep(self.backoff * attempt)
                continue
            except (OSError, http.client.HTTPException, ValueError) as e:  # connection, timeout, bad body
                self.errors[type(e).__name__] += 1
                attempt += 1
                time.sleep(self.backoff * attempt)
                continue
            a = r["answers"]
            scores = {ids[q]: a[q]["noul"] for q in qs} if ids else a["list"]["probabilities"]
            return {
                "decision": d["decision"],
                "variant": variant,
                "scores": scores,
                "cuts": cuts,
                "wall_ms": (time.perf_counter() - t0) * 1000,
                "latency_ms": r.get("latency_ms"),
                "truncated": bool(r.get("truncated")),
            }
        return None


def answers(
    reqs: Sequence[dict], wording: str, client: Client, revision: str, cache: Path
) -> tuple[dict, dict]:
    """Scores per (variant, decision id), from the cache or else the server. A request that stays unscored
    after a second pass stops the run, so a gap can never pass for a result."""
    have = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            have[rec["key"]] = rec
    got, todo = {}, []
    for d in reqs:
        for v in VARIANTS:
            k = key_of(revision, d["state"], build(d, v, wording)[0])
            if k in have:
                got[(v, d["decision"])] = have[k]
            else:
                todo.append((d, v, k))
    cached = len(got)
    if todo:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as f:
            for _ in range(2):  # the second pass retries what the first could not score
                left = []
                for d, v, k in todo:
                    rec = client.ask(d, v, wording)
                    if rec is None:
                        left.append((d, v, k))
                        continue
                    rec["key"] = k
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    got[(v, d["decision"])] = rec
                todo = left
    if todo:
        raise SystemExit(f"{len(todo)} Kev requests stayed unscored ({dict(client.errors)}); rerun to resume")
    stats = {
        "requests": len(got),
        "from_cache": cached,
        "cut_states": sum(1 for r in got.values() if r.get("cuts")),
        "errors": dict(client.errors),
    }
    return got, stats


def select(table: DecisionTable, folds: Mapping, vec: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """For each test fold, the scores of the wording with the best mean P@R on the other folds (ties keep the
    earlier wording)."""
    fold = fold_array(table, folds)
    fold_of = pd.Series(fold, index=table.decision_ids.astype(str).to_numpy()).groupby(level=0).first()
    per = {v: M.per_decision(table, s, "p@r") for v, s in vec.items()}
    chosen = {}
    for f in sorted(set(fold)):
        means = {}
        for v, p in per.items():
            means[v] = float(p[fold_of.reindex(p.index.astype(str)).to_numpy() != f].mean())
        chosen[int(f)] = max(means, key=means.get)
    return np.select([fold == f for f in chosen], [vec[v] for v in chosen.values()]), chosen


def kev_scores(
    table: DecisionTable,
    dataset: str,
    folds: Mapping,
    metrics: Sequence[str],
    *,
    url: str,
    revision: str,
    cache: Path,
    client: Client | None = None,
) -> tuple[np.ndarray, dict]:
    """Kev's score for every row of ``table``, and the run's record: the wording per fold, every wording's
    value."""
    reqs = requests(table, dataset)
    wording, id_col = TABLES[dataset][0], TABLES[dataset][3]
    got, stats = answers(reqs, wording, client or Client(url), revision, cache)
    ids = table.decision_ids.astype(str).to_numpy()
    names = table.df[id_col].astype(str).to_numpy()
    vec = {
        v: np.array([got[(v, d)]["scores"].get(c, np.nan) for d, c in zip(ids, names, strict=True)])
        for v in VARIANTS
    }
    missing = {v: int(np.isnan(s).sum()) for v, s in vec.items() if np.isnan(s).any()}
    if missing:
        raise SystemExit(f"Kev returned no score for some candidates: {missing}")
    sel, chosen = select(table, folds, vec)
    config = {
        "revision": revision,
        "wordings": list(VARIANTS),
        "chosen_by_fold": chosen,
        "variants": {
            v: {m: float(M.per_decision(table, s, m).mean()) for m in metrics} for v, s in vec.items()
        },
        **stats,
    }
    return sel, config
