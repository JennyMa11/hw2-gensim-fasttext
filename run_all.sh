#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/01_prepare.py
"$PYTHON_BIN" scripts/02_embeddings.py 2>&1 | tee results/embedding_training.log
"$PYTHON_BIN" scripts/03_embedding_analysis.py 2>&1 | tee results/embedding_analysis.log
"$PYTHON_BIN" scripts/04_classify.py 2>&1 | tee results/classification_training.log
"$PYTHON_BIN" scripts/05_verify_and_plot.py
"$PYTHON_BIN" scripts/08_epoch_diagnostic.py 2>&1 | tee results/epoch_diagnostic.log
"$PYTHON_BIN" scripts/06_report.py
