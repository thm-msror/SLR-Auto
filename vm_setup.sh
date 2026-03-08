#!/usr/bin/env bash
# run with -> ./setup.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-autoslr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' was not found in PATH."
  exit 1
fi

if conda env list | awk 'NR > 2 {print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Conda environment already exists: $CONDA_ENV_NAME"
else
  echo "Creating conda environment: $CONDA_ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
fi

if ! conda run -n "$CONDA_ENV_NAME" python --version >/dev/null 2>&1; then
  echo "Python not found in '$CONDA_ENV_NAME'. Installing python=$PYTHON_VERSION..."
  conda install -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
fi

echo "Upgrading pip tooling..."
conda run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
conda run -n "$CONDA_ENV_NAME" python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Installing Playwright browser binaries..."
conda run -n "$CONDA_ENV_NAME" python -m playwright install

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cat > "$ROOT_DIR/.env" <<'EOF'
# Azure GPT (used by screening, reading, and summaries)
GPT_ENDPOINT="https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
GPT_DEPLOYMENT="gpt-4o-kairos"
GPT_KEY="<your_azure_gpt_key_here>"
GPT_VERSION="2024-12-01-preview"
EOF
  echo "Created .env template at $ROOT_DIR/.env"
else
  echo ".env already exists. Skipping creation."
fi

echo
echo "Setup complete."
echo "Next steps:"
printf '  1) conda activate "%s"\n' "$CONDA_ENV_NAME"
printf '  2) Edit .env and add your real API keys\n'
printf '  3) python main.py\n'
