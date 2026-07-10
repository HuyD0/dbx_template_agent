# Troubleshooting

The errors new users actually hit, with the actual fix. Notebook 01's
preflight cell is designed to surface most of these in the first 5 minutes.

## Auth & workspace

**`default auth: cannot configure default credentials`** — the CLI/SDK found
no identity. Run `databricks auth login --host https://<your-workspace>`, or in
a notebook this never happens (ambient auth). Check `~/.databrickscfg` exists
and `DATABRICKS_CONFIG_PROFILE` points at the right profile.

**`403 PERMISSION_DENIED` calling a serving endpoint** — your user/SP lacks
**CAN QUERY** on that endpoint (Serving → endpoint → Permissions). On some
accounts, pay-per-token foundation models are rate-limited to zero for certain
principals — test the endpoint in the Serving UI playground; if it fails there
too, it's the endpoint/entitlement, not your code.

**`404` / `RESOURCE_DOES_NOT_EXIST` for the endpoint** — the endpoint name in
`config.py` doesn't exist in THIS workspace. List what you have:
`databricks serving-endpoints list`. Foundation-model names differ by region.

## Unity Catalog

**`CATALOG_DOES_NOT_EXIST` / permission denied on schema** — bundles cannot
create catalogs. Ask an admin for `USE CATALOG` + `USE SCHEMA` + `CREATE`
on your catalog/schema, or point `config.py` at ones you own
(`catalog.schema` must exist before notebook 01 registers the prompt).

**Prompt/model registration fails with `INVALID_PARAMETER_VALUE`** — the name
isn't a valid three-level UC name. Lowercase snake_case, three parts:
`main.my_agent.my_agent_system`.

## Evals & judges

**Judge calls time out or 429** — judges run per-row; a big eval set on a
throttled endpoint will crawl. Shrink the set, or set
`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=1` to serialize scorer traffic.

**`correctness/mean` missing from results** — rows lack
`expectations.expected_response`; the Correctness judge silently skips rows it
can't ground. Check the jsonl.

**Gate REFUSED: estimated cost exceeds budget** — working as intended (cost
fails closed). Raise `DAILY_BUDGET_USD` in `config.py`, or shrink the eval
set, or wait for tomorrow's budget.

**Judge scores look suspiciously high** — is your judge the same model family
as the agent? Same-family judges over-score their siblings. Use a different
family (that's why init asked for a separate judge endpoint).

## Bundle & CI

**`databricks bundle validate` fails with a variable error** — you edited
`databricks.yml` and removed a variable a resource still references. The
resource files under `resources/` use `${var.*}` — keep them in sync.

**Deployed notebook can't import the project package** — the `sys.path.insert`
cell at the top of each notebook must run first; it makes `../src` importable
relative to the notebook's deployed location. Run notebooks top-to-bottom.

**`%pip install` then `ModuleNotFoundError`** — you skipped
`%restart_python`. The kernel must restart before new packages import.

## Traces

**"My trace isn't in the experiment"** — you logged before calling
`config.setup_mlflow("agent")`, so it went to the notebook's default
experiment. Re-run the setup cell first; check the experiment named in the
cell output.

**Trace has no TOOL spans** — the model answered without calling tools
(check `tools_used` in the result). If it SHOULD have called one, look at the
system prompt: the "use the calculator for ANY arithmetic" instruction is
what drives tool choice.
