# dbx_template_agent — Claude Context

## Project Purpose
This is a **Databricks asset bundle template** for a tool-calling agent with a full MLflow eval lifecycle. It's designed as a teaching template — 8 progressive notebooks walk users through: build → trace → experiment → evaluate → gate/deploy → monitor.

## Current Status
- **Bundle**: Parameterized `databricks.yml` with dev/prod targets
- **Notebooks**: 8 teaching notebooks (01-08) deployed as workspace sources
- **Evals**: Complete eval harness with gate, scorers, thresholds
- **Templates**: Serving endpoint resource commented out (needs model v1 from nb 06)
- **Template readiness**: ~95% — missing: notebook resource definitions, SETUP.md docs

## Architecture Decisions
- **Prompts in MLflow registry** (not hardcoded) — enables alias-based promotion
- **Eval sets as run inputs** — lineage + versioning via `dataset_version`
- **Fail-closed cost gate** — refuses to exceed `DAILY_BUDGET_USD`
- **Notebooks are source files** — not deployed as job tasks (users run manually)
- **Service principal for prod** — commented in databricks.yml, requires setup

## Pre-requisites for Users
Before running `bundle init`:
1. Pre-create UC catalog and schema (or use defaults: `main.dbx_template_agent`)
2. Pre-create two serving endpoints (or use Databricks-managed ones)
3. Set workspace host in `databricks.yml` (currently hardcoded)

## Template Conversion Checklist
When converting this back to a true template, verify:
- [ ] `databricks bundle validate && deploy --target dev` succeeds
- [ ] Notebook 01 runs end-to-end in deployed workspace
- [ ] Add SETUP.md for step-by-step user onboarding
- [ ] Consider adding notebook resource definitions to `resources/*.yml`
- [ ] Test as fresh clone (simulate new user experience)
- [ ] Tag and release as v0.1.0 template

## Convention: When Touching the Bundle
- Always run `databricks bundle validate` before committing
- Update `pyproject.toml` version when releasing a new template version
- Keep `databricks.yml` comments clear about pre-requisites
- Notebooks use `sys.path.insert(0, "../src")` for package imports

## Key Files
- `databricks.yml` — bundle config with parameterized vars
- `src/dbx_template_agent/config.py` — single source of truth for all settings
- `resources/*.yml` — job and serving endpoint resource definitions
- `evals/thresholds.yaml` — cost and quality gate floors
- `notebooks/` — the 8-phase teaching curriculum

## CI/CD (mirrors the agent-eval repo's split)
Three workflows in `.github/workflows/`, same philosophy as
`~/Projects/Python/agent-eval`: hermetic checks auto-run; anything that spends
money is **manual dispatch only**.

| Workflow | Trigger | Needs secrets? | What it does |
|---|---|---|---|
| `validate.yml` | every PR + manual | no | ruff lint/format, offline pytest, `bundle validate` (schema only, fake host) |
| `secret-scan.yml` | every PR/push + manual | no | gitleaks over full history |
| `deploy.yml` | **manual dispatch only** | yes | `bundle deploy` + optionally `bundle run` the eval gate (spends real compute + judge tokens) |

### Driving CI/CD from Claude mobile
Everything is triggered through `gh`, so any Claude session (mobile included)
can run full CI/CD without a local checkout:

```bash
gh workflow run deploy.yml -f run_eval_gate=true   # deploy + eval gate
gh run list --workflow=deploy.yml --limit 3         # check status
gh run watch                                        # tail a run
gh run view <id> --log-failed                       # debug a failure
```

Repo secrets required (set once): `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
(service principal token; prefer OIDC federation long-term). The `deploy.yml`
job uses the `production` GitHub environment — add a required-reviewer rule
there if you want an approval tap before money is spent.
