#!/usr/bin/env bash
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
# FANAR (used by screening step)
FANAR_API_KEY=your_fanar_api_key_here
FANAR_BASE_URL=https://api.fanar.qa/v1

# Gemini (used by full-paper reading and summaries)
GEMINI_API_KEY=your_gemini_api_key_here
EOF
  echo "Created .env template at $ROOT_DIR/.env"
else
  echo ".env already exists. Skipping creation."
fi

echo
echo "Setup complete."
echo "Next steps:"
echo "  1) conda activate \"$CONDA_ENV_NAME\""
echo "  2) Edit .env and add your real API keys"
echo "  3) python main.py"
