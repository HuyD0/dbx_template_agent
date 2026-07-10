#!/usr/bin/env bash
# One-time setup: keyless (OIDC) auth from GitHub Actions to Databricks.
#
# Creates a Databricks service principal, adds a federation policy that trusts
# GitHub's OIDC issuer for THIS repo's `production` environment, grants the SP
# access to the workspace, and stores its client id as a GitHub Actions
# variable. After this, deploy.yml authenticates with a short-lived token
# minted per run — no DATABRICKS_TOKEN secret to create, rotate, or leak.
#
# Run interactively (it opens a browser for the account-console login):
#   bash scripts/setup_github_oidc.sh
set -euo pipefail

ACCOUNT_ID="617a10e3-e106-4d01-bc34-524981fc9683"
WORKSPACE_ID="7405609799238491"
WORKSPACE_HOST="https://adb-7405609799238491.11.azuredatabricks.net"
REPO="HuyD0/dbx_template_agent"
GH_OWNER_URL="https://github.com/HuyD0"
SP_NAME="github-dbx-template-agent"
PROFILE="account"

command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }

echo "==> 1/5 Log in to the Databricks account console (profile: $PROFILE)"
databricks auth login \
  --host https://accounts.azuredatabricks.net \
  --account-id "$ACCOUNT_ID" \
  --profile "$PROFILE"

echo "==> 2/5 Create (or reuse) service principal '$SP_NAME'"
EXISTING=$(databricks account service-principals list -p "$PROFILE" -o json |
  jq -r --arg n "$SP_NAME" '.[] | select(.displayName == $n) | .id' | head -1)
if [[ -n "$EXISTING" ]]; then
  SP_ID="$EXISTING"
  CLIENT_ID=$(databricks account service-principals get "$SP_ID" -p "$PROFILE" -o json | jq -r '.applicationId')
  echo "    reusing existing SP id=$SP_ID"
else
  SP_JSON=$(databricks account service-principals create \
    --display-name "$SP_NAME" --active -p "$PROFILE" -o json)
  SP_ID=$(echo "$SP_JSON" | jq -r '.id')
  CLIENT_ID=$(echo "$SP_JSON" | jq -r '.applicationId')
  echo "    created SP id=$SP_ID"
fi
echo "    client id (applicationId) = $CLIENT_ID"

echo "==> 3/5 Federation policy: trust GitHub OIDC for repo=$REPO, environment=production"
databricks account service-principal-federation-policy create "$SP_ID" \
  -p "$PROFILE" --json "{
    \"oidc_policy\": {
      \"issuer\": \"https://token.actions.githubusercontent.com\",
      \"audiences\": [\"$GH_OWNER_URL\"],
      \"subject\": \"repo:$REPO:environment:production\"
    }
  }"

echo "==> 4/5 Grant the SP USER access to workspace $WORKSPACE_ID"
databricks account workspace-assignment update "$WORKSPACE_ID" "$SP_ID" \
  -p "$PROFILE" --json '{"permissions": ["USER"]}'

echo "==> 5/5 Store the client id as a GitHub Actions variable (it is an id, not a secret)"
gh variable set DATABRICKS_CLIENT_ID --body "$CLIENT_ID" --repo "$REPO"
gh variable set DATABRICKS_HOST --body "$WORKSPACE_HOST" --repo "$REPO"
# The PAT-based secret is no longer needed; remove it if present.
gh secret delete DATABRICKS_TOKEN --repo "$REPO" 2>/dev/null || true

echo
echo "✅ Done. deploy.yml now authenticates keylessly via OIDC."
echo "   Test it:  gh workflow run deploy.yml -f run_eval_gate=true && gh run watch"
