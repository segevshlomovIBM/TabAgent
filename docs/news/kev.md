---
title: "TabAgent beside Kev"
description: "The released Kev decision models, run as one more baseline under the paper's protocol on the five public set-retrieval tables."
kind: "News"
lead: "Kev as released, with no fine-tuning, through the same harness, text, folds and metrics as the paper's baselines."
---

# TabAgent beside Kev

*November 2026.* [Kev](https://github.com/jaredpalmer/kev) is a self-hosted decision model: given an agent's
state and a set of options, it scores each option. It could therefore make an agent's closed-set decision without a
head trained on that agent's runs. We ran its three released sizes, Kev 1.0 at 4B, 9B and 27B (tag `v1.0` on
Hugging Face; the 27B model card calls it "v2"), as one more baseline on the paper's five public set-retrieval tables.
Kev's checkpoints are its authors' fine-tunes of Qwen models; we did not fine-tune them.

**In short.** On τ²-bench, TabAgent's head, running on a CPU, beats the largest Kev model, a 27B model that needs an
80 GB GPU, by 18 to 37 points of P@R on retail, telecom and the combined table, and is level with it on airline. It is
also faster per decision. Kev-27B is ahead on SkillsBench skill selection; the smaller Kev-9B and Kev-4B are level
with TabAgent there.

## Results

P@R, percent. *Paper* is TabAgent's real-only value in Table 1. *Recomputed* is the same head rerun for this
comparison, on the decisions Kev was scored on.

| Table | Decisions (tasks) | Paper | Recomputed | Kev-27B | Kev-9B | Kev-4B |
|---|---|---:|---:|---:|---:|---:|
| τ² airline | 91 (24) | 72 | 72.9 | 69.4 | 63.4 | 56.5 |
| τ² retail | 153 (54) | 75 | 77.7 | 60.7 | 52.3 | 58.6 |
| τ² telecom | 82 (61) | 85 | 84.2 | 47.7 | 43.3 | 47.3 |
| τ² combined | 326 (139) | 77 | 77.3 | 45.7 | 42.5 | 45.2 |
| SkillsBench skill selection | 87 (87) | 80 | 79.3 | 90.6 | 79.8 | 78.8 |

Kev minus the recomputed TabAgent, P@R points, paired over tasks with a 95% interval. **Bold** means Holm-adjusted
p below 0.05.

| Table | Kev-27B | Kev-9B | Kev-4B |
|---|---|---|---|
| τ² airline | −3.1 [−12.8, +6.6] | −6.9 [−17.2, +3.6] | **−13.5 [−21.9, −5.7]** |
| τ² retail | **−18.0 [−23.5, −13.0]** | **−24.7 [−30.2, −19.9]** | **−19.4 [−25.8, −12.5]** |
| τ² telecom | **−36.7 [−41.3, −32.2]** | **−42.3 [−50.2, −33.3]** | **−38.0 [−46.0, −28.7]** |
| τ² combined | **−34.5 [−38.2, −31.0]** | **−38.7 [−43.2, −33.6]** | **−34.6 [−39.4, −29.5]** |
| SkillsBench skill selection | **+11.4 [+6.2, +16.4]** | +0.5 [−4.2, +5.2] | −0.5 [−5.5, +4.7] |

Recall@9, percent. *Paper* is the real-only value (Table 11 for τ², Table 1 for skill selection).

| Table | Paper | Recomputed | Kev-27B | Kev-9B | Kev-4B |
|---|---:|---:|---:|---:|---:|
| τ² airline | 98 | 97.5 | 93.2 | 96.4 | 95.8 |
| τ² retail | 96 | 95.7 | 92.6 | 85.1 | 85.0 |
| τ² telecom | 100 | 99.7 | 92.2 | 91.0 | 89.1 |
| τ² combined | 97 | 97.0 | 79.4 | 72.0 | 72.9 |
| SkillsBench skill selection | 91 | 90.1 | 97.5 | 94.8 | 95.8 |

## Speed and hardware

Per decision on τ² combined (38 candidates). TabAgent's time is its full first call, which also embeds the text,
over 69 decisions. Kev's is a new state in the wording behind its P@R, one request at a time, over a pass of 240
decisions.

| Method | Hardware | Median ms | GPU memory after loading |
|---|---|---:|---:|
| TabAgent | AMD EPYC 7763, 4 CPU threads | 53.7 | none |
| Kev-27B | A100 80 GB / H100 80 GB | 638 / 268 | 64,031 / 66,717 MiB |
| Kev-9B | A100 40 GB / H100 80 GB | 216 / 95 | 22,725 / 25,641 MiB |
| Kev-4B | A100 40 GB / H100 80 GB | 141 / 70 | 15,297 / 17,999 MiB |

The hardware differs, so these are not like-for-like speeds. Kev's one-question listwise wording is faster (Kev-27B:
131 ms on the A100, 65 ms on the H100) and scores lower on this table (41.9 P@R for Kev-27B). The classifier pass
alone, the number the paper reports, takes 3.2 ms here. TabAgent's times come from
`python -m experiments.efficiency.latency --datasets tau2/all-domains --n 200 --repeats 5`; Kev's from timing the
same requests `kev-zs` sends, against its server.

## How it was run

- **The paper's harness.** `kev-zs` is a method of `experiments.baselines.run`, which loads each table with its
  pinned task folds (`tabagent.datasets.folds`) and writes the same per-decision results as for DSR and the LLM
  baselines. Every decision was scored, and the decision counts match TabAgent's on every row.
- **The baselines' text.** Kev read the preset's views, as the paper's baselines do: on τ² the customer's instruction
  and the tool name, on skill selection the task and each skill's name and description.
- **No choice saw its test fold.** Six question wordings per model were fixed before any score met a label. Each
  test fold used the wording with the best mean P@R on the other four folds.
- **The paper's statistics.** Tied scores count at their exact expected value. Intervals come from a paired bootstrap
  over tasks with 2,000 resamples. Holm runs over the 57 comparisons of each metric: 3 Kev models; TabAgent
  real-only and +synth and both DSR-FT recipes; 5 tables. Its p-values use 20,000 resamples.
- **Full precision.** We served Kev with its probabilities unrounded. The stock server rounds them to 4 decimals,
  which moves P@R by at most 0.1 here.
- **TabAgent recomputed** with the cell logic of
  `python -m experiments.reproduce --download --cells tau2-airline tau2-retail tau2-telecom tau2-combined skill-selection`,
  on x86-64 (AMD EPYC). τ² retail gives 77.7 there and 75.4 on an Apple M1 Max.

To rerun one row, serve Kev from [its repository](https://github.com/jaredpalmer/kev) (Kev-27B on an 80 GB GPU;
Kev-9B and Kev-4B ran on 40 GB A100s), then score the table:

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b --port 8008
python -m experiments.baselines.run --dataset tau2/airline --method kev-zs --model jaredpalmer/kev-27b --download
```

Revisions: TabAgent `1545334`; Kev code `5e42a7a`; `jaredpalmer/kev-27b` `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf`,
`kev-9b` `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0`, `kev-4b` `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`.

## Limits

- **"Zero-shot" is only partly verified.** None of the five benchmarks appears by name in Kev's model cards,
  release notes or training manifest. The manifest lists generated records (`synthetic-v1/tool_routing`,
  `synthetic-v2/agent_sessions`) that can't be audited, and neither can the Qwen base models' training data.
- **Names only on τ².** Kev reads tool names, as the paper's baselines do. Given tool descriptions, arguments and
  domain as well, Kev-27B reaches 65.7 on τ² combined, still below TabAgent's 77.3; that run is not comparable with
  the tables above.
- **Skill selection.** Kev reads each skill's description. A skill's name appears in the task text for 40 of the 232
  relevant skills (17.2%) and 9 of the 16,733 others (0.1%), so a literal match explains part of Kev-27B's lead.
  Whether Kev's training data overlaps SkillsBench cannot be fully audited.
- **Not run.** Kev fine-tuned on the agent's runs; AppWorld, whose trajectories are not released.
