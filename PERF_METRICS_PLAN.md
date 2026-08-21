# Issue #97 — efficiency metrics: plan

Draft for issue #97. Not a repo doc, not committed — paste target.

Revised after a four-model review panel (Fable 5, Opus 4.8, Gemini 3.1 Pro,
Gemini 3.7 Flash). Where a section changed, the reason is stated inline.

## Position

Most of what #97 asks for is already collected. The work splits three ways:

1. **Bugs.** The API harness reports the wrong token counts today.
2. **Collection.** Measurements that are gone if the run does not capture them.
3. **Surfacing.** Numbers already in the record that never reach `ResultRow`.

The rule everywhere below: **store measurements, derive judgments.** A number
goes on the row only if it cannot be recomputed from what is already there.

### Where the issue's proposal changes

**No `PerformanceMonitor` wrapper — but the live capture point it implies is
real, and does not exist today.** A module that only sits between
`AgentResult` and `ResultRow` restating them would earn nothing. But the panel
showed that every place the plan originally wanted to stamp a clock is
post-hoc: `devops_bench/agents/api/agent.py:145` is inside
`_fold_with_extraction_errors`, which runs *after* the loop returns, and all
three CLI parsers are pure functions over a finished transcript. So the issue
was right that something has to wrap execution. It is one thing, in one place:
the dispatch call inside `run_tool_loop`. See 2b.

**Host CPU / memory stays out, with one exception.** It measures the runner,
not the model; two runs of one model on different hardware would score
differently. The exception the panel raised is fair: for a locally-hosted model
(Ollama, vLLM) memory footprint *is* the deployment cost, the direct analogue
of an API bill. If and when the benchmark runs local weights, capture it then,
gated on that. Not now, and never on a leaderboard comparing hosted APIs.

**Cost: derived, but the rate card must be immutably date-versioned.** Three
of four reviewers pushed back on pure derivation and they are right about the
failure: with one render-time rate card, a Q3 price cut retroactively rewrites
what a Q1 run cost, and a rate card for a deprecated model eventually cannot be
recovered at all. Storing `cost_usd` is still wrong — it freezes one
interpretation and makes any pricing correction a backfill. The middle is to
stamp `rate_card_version` on the row and keep rate cards as dated, immutable
files. Then "what did this cost when we ran it" and "what would it cost today"
are both computable, and neither silently overwrites the other.

## What already exists

| Thing | Where | Caveat |
|---|---|---|
| Six-bucket token schema | `devops_bench/agents/result.py`, `devops_bench/results/row.py` | schema only — see below |
| Whole-run latency | `AgentResult.latency` -> `ResultRow.latency_sec` | means two different things — see 2f |
| Tool trajectory | `AgentResult.trajectory` | does not travel onto `ResultRow` |
| Per-call tool status | all four parsers set `completed`/`error` | plus `interrupted`, and residual `called` |
| Model, harness, augmentation, setup | `Manifest` | |

The six buckets are a schema, not coverage. `extract_tokens`
(`devops_bench/agents/api/agent.py:160`) emits only `prompt_tokens` /
`candidates_tokens` / `total_tokens`, so every API-harness row has `cached`,
`cache_write` and `reasoning` as `None` and cache-discounted or
reasoning-priced cost cannot be computed for API runs at all.

## 1. Bugs

### 1a. API token counts were the last turn only — FIXED

`devops_bench/agents/api/agent.py` read usage off `LoopResult.response`, the
final provider response, and reported it as the run's usage. Each turn is
billed separately and the whole conversation is re-sent every time, so a
multi-turn run reported a fraction of what it was billed, with the shortfall
growing in turn count.

`run_tool_loop` now retains every per-turn response in `LoopResult.responses`
and the harness sums them via `sum_tokens`.

### 1b. Only three of six token buckets are extracted

Same function, separate defect. Widening it needs the inclusive/exclusive
semantics right per provider — Google's `prompt_token_count` already includes
cached tokens, Anthropic's `input_tokens` does not, so a naive sum
double-counts for one and undercounts for the other. Its own change.

## 2. Collection — gone if we do not capture it

### 2a. Per-turn records

Add `turns` to `LoopResult` (`devops_bench/models/utils/loop.py`): per provider
call, the token buckets and the seconds that call took.

Gives model-call count (the issue's "API call count" — `len(trajectory)` is
tool calls, a different number), the context-growth curve, and where a run
stopped making progress. On an agentic loop the whole history is re-sent every
turn, so input tokens grow roughly quadratically in turn count; a run total
hides that entirely.

**Not API-only.** The panel found that openclaw already emits one
`model.completed` event per model call with usage attached, and antigravity's
DB stores one usage record per turn. For those two this is a parser change, not
a collection change, and `model_turns` can cover three harnesses rather than
one.

### 2b. Per-tool-call timing — capture at dispatch, not at the fold

**This section was wrong in the first draft and is the panel's main finding.**

Two nullable fields on `ToolCall` (`devops_bench/agents/result.py:36`):

```python
started_at: float | None = None   # seconds since agent start, monotonic
duration_sec: float | None = None
```

The capture point is the dispatch call inside `run_tool_loop`
(`devops_bench/models/utils/loop.py:127-141`) — the only place in the API path
where a tool call is live. The values then have to be threaded through
`contents` so `fold_trajectory` can attach them, since the fold is what builds
`ToolCall`.

**The three CLI harnesses cannot do this post-hoc.** Their parsers read a
finished transcript — gemini_cli's stream events, openclaw's `events.jsonl`,
antigravity's session DB — and none of those event shapes carries a timestamp
or duration. Antigravity is worse: its parser replays `$rewindTo` rewinds, so
reconstructed per-call wall time would not mean anything even if timestamps
existed.

So: **API harness first.** Before extending to the CLI harnesses, spike each
one against live CLI output (not the test fixtures) to see whether the tool
actually emits per-event times. If it does not, `tool_wait_sec` and everything
derived from it are API-only columns and the plan must say so rather than ship
a column that is null on most rows.

Unlocks, for the harnesses that can produce it: `tool_wait_sec`, slowest tool
per task, and the split that matters here — most wall time on a Kubernetes task
is the cluster, not the model.

### 2c. Terminal reason

One enum on `AgentResult`: `completed | max_turns | timeout | error`.

Stronger than the first draft claimed. A timeout does change the row (it routes
through `AgentResult.errored` to `status="failed"`), but **cap exhaustion is
recorded nowhere at all** — `run_tool_loop` logs a warning
(`devops_bench/models/utils/loop.py:143`) and returns an ordinary `LoopResult`.
A capped run is today indistinguishable from a completed one in the data. The
implementation therefore has to surface cap exhaustion on `LoopResult` first.

### 2d. Provider retry and backoff time

`devops_bench/models/gemini.py:263` retries 429/503 up to five times with
exponential backoff. That wait sits inside `latency` and inside any think-time
figure derived from it.

Two constraints the first draft missed. The retry loop lives inside the client
adapter with no channel back to the agent layer, so this needs a stats hook on
`LLMClient` surfaced through `LoopResult`. And gemini is the only in-tree
adapter that retries — the Anthropic and OpenAI SDKs retry internally and
invisibly, so the field reads "zero retries" for them. Mark it provider-scoped
and `None` where unknown, rather than letting a cross-provider comparison treat
absent as zero.

### 2e. The caps that were in effect

`max_turns` and `timeout_sec` (`devops_bench/agents/config.py:112`) onto
`Manifest`. "Used 40 turns" means nothing without the cap.

`max_turns` is read only by the API agent — no CLI harness consumes it. Stamp
it nullable and only where it applies, or a CLI arm's manifest advertises a
budget that never bound it.

### 2f. `latency` means two different things

The API harness sets `AgentResult.latency` from `loop_result.latency`, which is
inference time only. No CLI harness sets it at all — `AgentHarness.run`
(`devops_bench/agents/base.py:136-137`) fills it when `_execute` left it zero,
and that span covers the whole of `_execute`: the agent subprocess plus
transcript parsing plus openclaw's post-run `oc sessions` / export subprocesses
plus antigravity's DB flush polling.

**Do not rename `latency_sec` to `wall_sec`.** A rename relabels every
historical API row as wall clock when it holds inference-only time. Add
`wall_sec` as a new column and document `latency_sec`'s per-harness meaning.

**Do not derive `inference_sec` by subtraction.** All three code-reading
reviewers rejected this independently, which is the strongest agreement the
panel produced. `wall_sec - tool_wait_sec - overhead` dumps network transit,
serialization and every unaccounted harness delay into the model's number, so a
CLI-tested model reads as slower than it is. `inference_sec` is measured
directly or it is not reported. Today that means API-only.

### 2g. Multiple iterations per task

`devops_bench/results/normalize.py:307` hardcodes `iteration=0`. One run per
task, so variance cannot be derived after the fact.

The first draft put this in "later". The panel argued, and I agree, that it
should move up: on a leaderboard where top models are within a few points,
single-run data reorders on rerun, and every percentile in section 4 is a
percentile over one sample. Own issue, but a blocker on publishing rankings
rather than a nice-to-have.

## 3. Surfacing — collected, never reaches the row

The trajectory does not travel on `ResultRow`, so these are computed at
normalize time and stored. All nullable.

| Field | From | Harness coverage |
|---|---|---|
| `tool_calls` | `len(trajectory)` | all four |
| `tool_errors` | status `error` **or** `interrupted` | all four |
| `model_turns` | 2a | api, openclaw, antigravity |
| `terminal_reason` | 2c | all four |
| `tool_wait_sec` | 2b | api (others pending spike) |
| `inference_sec` | 2f | api only |
| `wall_sec` | 2f | all four |
| `rate_card_version` | manifest | all four |

`tool_errors` must count `interrupted` — antigravity emits it for calls still
pending at EOF. Calls left at `called` (never resolved) are neither success nor
error; count them separately or document the exclusion, because they cluster on
exactly the truncated runs 2c exists to identify.

**Dropped from the first draft:** `mutating_calls` and
`time_to_first_mutation_sec`. Nothing in the repo classifies a tool as
mutating, and for a harness whose agent drives a generic shell tool the
mutation is in the arguments (`kubectl delete ...`), not the tool name. Both
columns would ship null or, worse, be computed differently per harness. If they
are wanted, a tool classifier is a deliverable in its own right, not a
parenthetical.

Not stored, computed at report time: cost, cache hit rate, tokens per turn,
inference share of wall clock. Every input is on the row.

## 4. Report output

Ordered for someone choosing between models that score the same.

1. **Total suite cost divided by total tasks solved.** Not per-task cost
   averaged over solved tasks — that is undefined when a model solves nothing
   and is pure survivor bias otherwise. Report **wasted cost on failed tasks**
   beside it.
2. **Score against cost as a frontier** — which models are not dominated.
3. **Inference time and tool wait, reported separately.** Wall clock does not
   go at the top: the plan's own argument is that the cluster dominates it, so
   ranking on it ranks provisioning luck. Keep wall clock as a diagnostic.
4. **Turns to solve**, median.
5. **Tool error rate.**
6. **Failure profile** — of the failures, how many hit the turn cap or the
   timeout versus finished and got it wrong. This is where 2c pays off.

Nothing in 1-3 is publishable as a ranking until 2g lands. Until then it is
diagnostic output with the sample size stated.

## Phasing and PR split

Five separate PRs. They share a topic, not code; only PR 4 depends on PR 1.

| PR | Scope | Depends on |
|---|---|---|
| 1 | Per-turn token accumulation (1a) | — **open** |
| 2 | Six-bucket extraction for the API harness (1b) | — |
| 3 | Terminal reason (2c), incl. cap exhaustion on `LoopResult` | — |
| 4 | Per-turn records (2a) + `model_turns` | 1 |
| 5 | `tool_calls` + `tool_errors` on the row (3) | — |

Blocked, no PR yet:

- **Tool timing (2b).** Needs the dispatch-site design, then a per-harness
  spike against live CLI output.
- **`inference_sec` (2f).** Needs direct instrumentation or an explicit
  API-only scope.
- **Mutating-call classification.** Needs a classifier that does not exist.
- **Report output (section 4).** Gated on 2g.
- **Multiple iterations (2g).** Own issue.

Phase 0, before any of the blocked items: audit what `latency` actually spans
per harness — does it include infra provisioning, verification, the GEval
judge? `base.py:136` says it spans all of `_execute`, but what `_execute`
covers differs per harness. If it includes provisioning, `wall_sec` as defined
above is measuring our infrastructure and needs another boundary stamped.

## Open questions

- Does the ingest validator reject unknown row fields? If it does, the
  dashboard schema has to move before PR 5, and that lives in another repo.
- Where do dated rate cards live, and who updates them?
- Is `wall_sec` worth adding for the API harness at all, given it currently has
  no whole-run measurement distinct from inference time?
