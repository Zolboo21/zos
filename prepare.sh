#!/bin/bash
set -e

# ── 1. Ensure uv is installed ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv
fi

# ── 3. Sync dependencies from pyproject.toml ──────────────────────────────────
echo "Syncing dependencies..."
uv sync

# ── 4. Download dataset ───────────────────────────────────────────────────────
mkdir -p ./notebook

if [ -f "./notebook/dataset.csv" ]; then
    echo "Dataset already present, skipping download."
else
    echo "Downloading dataset from Hugging Face..."
    uv run python - <<'EOF'
from datasets import load_dataset

ds = load_dataset("Noysaa/data_scientist_task", split="train")
ds.to_pandas().to_csv("./notebook/dataset.csv", index=False)
print(f"Saved {len(ds)} rows to ./notebook/dataset.csv")
EOF
fi
