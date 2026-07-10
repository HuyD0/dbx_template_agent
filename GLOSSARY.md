# Glossary

**trace** — the record of ONE request through the agent: a tree of spans with
inputs, outputs, timing, and attributes. The unit of observability.

**span** — one step inside a trace. Typed: `AGENT` (the root), `CHAT_MODEL`
(a model call; carries token `usage`), `TOOL` (a tool execution). The span
structure your code emits is its **trace contract** — scorers rely on it.

**experiment / run** — an MLflow *experiment* is a folder of *runs*; a run is
one measured execution with tags, params, metrics, and linked inputs. This
project keeps one experiment per concern: `/agent`, `/experiments`, `/evals`,
`/monitoring`.

**tags vs params vs metrics** — tags: what you *filter* runs by (git commit,
dataset version). Params: the *knobs* this run turned (temperature, prompt
version). Metrics: what *happened* (scores, cost, latency).

**eval** — not a score: a versioned triple of **(dataset, scorers,
thresholds)**. Change any leg and it's a different eval.

**scorer** — any function that grades one eval row. *Deterministic* scorers
(regex, span checks, ceilings) are exact and free; use them wherever possible.

**LLM judge** — a scorer that is itself a model, for criteria needing language
understanding (correctness, relevance). Must be a *different model family*
than the agent (self-enhancement bias) and *calibrated against human labels*
before it gates anything.

**judge alignment / calibration** — measuring (and optimizing) a judge's
agreement with human labels on a holdout. The judge's own eval.

**reference-free judge** — a judge needing no ground-truth answer (e.g.
relevance), so it can score *live traffic* — the monitoring workhorse.

**assessment / feedback** — a label attached to a trace (`mlflow.log_feedback`):
human 👍/👎 or a judge verdict. Human feedback mined from traces is where the
best new eval rows come from.

**gate** — the eval run whose threshold floors (quality AND cost) decide
whether a candidate may be promoted. Exit code 0 = promotable.

**champion / challenger** — the incumbent vs the candidate, compared on the
same eval set. Promotion = moving the `@champion` alias.

**alias** — a movable pointer to a model/prompt version (`@production`,
`@champion`). Downstream code resolves aliases and never hardcodes versions;
rollback is moving the pointer back.

**prompt registry** — MLflow's versioned store for prompts. A registered
prompt gives every score an exact prompt version, not a guess.

**dataset lineage** — logging the eval set as a run *input*
(`mlflow.log_input`) so the run↔dataset edge is traversable — a tag alone
isn't.

**Databricks Asset Bundle (DAB)** — workspace resources (jobs, experiments,
endpoints) described as versioned YAML, deployed with `databricks bundle
deploy`. This project is one.

**Unity Catalog (UC)** — Databricks governance; everything named lives at
`catalog.schema.name` (lowercase snake_case).

**Agent Bricks** — Databricks' managed, config-driven agent product. Notebook
08 pits one against your hand-coded agent — using your eval set as the referee.
