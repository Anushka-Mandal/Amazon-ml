#!/usr/bin/env bash
# End-to-end: data -> normalisation -> blocking -> features -> model -> output/*.tsv
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY learn_translit.py          # native-script dictionaries learned from training labels
$PY prepare.py train           # normalise train sources
$PY prepare.py test            # normalise test sources
$PY blocking.py train          # candidate generation on train
$PY train.py                   # features + LightGBM + decision-rule tuning (validation F0.5)
$PY blocking.py test           # candidate generation on test
$PY predict.py                 # score test candidates, write output/
