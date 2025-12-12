#!/bin/bash
set -e

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

echo "Running data processing..."
python 01-data-preprocessing.py

#echo "Running notebook: Label Exploration..."
#jupyter nbconvert --to notebook --execute --output-dir output notebook/01-data-exploration.ipynb

#echo "Running notebook: Label Analysis..."
#jupyter nbconvert --to notebook --execute --output-dir output notebook/02-label-analysis.ipynb

echo "Running model training..."
python 02-train.py

echo "Running model evaluation..."
python 03-evaluation.py

echo "Running model inference..."
python 04-inference.py

echo "Running baseline comparison..."
python 05-baseline-comparison.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"