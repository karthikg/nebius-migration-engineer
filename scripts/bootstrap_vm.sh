#!/usr/bin/env bash
# Bootstrap the Migration Engineer on a fresh Nebius Compute VM (Ubuntu 22.04/24.04).
# Usage (on the VM):
#   export REPO_URL=https://github.com/karthikg/nebius-migration-engineer.git
#   curl -fsSL https://raw.githubusercontent.com/karthikg/nebius-migration-engineer/main/scripts/bootstrap_vm.sh | bash
# Or clone first and run: bash scripts/bootstrap_vm.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/karthikg/nebius-migration-engineer.git}"
APP_DIR="${APP_DIR:-$HOME/nebius-migration-engineer}"

echo "==> Installing system packages"
sudo apt-get update -y
sudo apt-get install -y git python3-venv python3-pip

if [ ! -d "$APP_DIR/.git" ]; then
  echo "==> Cloning $REPO_URL"
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

PY="$(command -v python3.13 || command -v python3.12 || command -v python3)"
echo "==> Creating venv with $PY"
"$PY" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e .

if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "!! Edit $APP_DIR/.env and set NEBIUS_API_KEY, then verify with:"
  echo "   $APP_DIR/.venv/bin/tf-migrate preflight"
else
  echo "==> .env already present"
fi

echo ""
echo "==> Done. Try:"
echo "   cd $APP_DIR && ./.venv/bin/tf-migrate run workloads/summarization.yaml"
