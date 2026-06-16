#!/usr/bin/env bash
set -euo pipefail

python test_IQAG.py \
  --dataset-config configs/datasets.json \
  --datasets cnnspot genimage \
  --checkpoint checkpoints/IQAG_best.pth \
  --clip-checkpoint pretrained/ViT-L-14.pt \
  --iqa-csv-dir /path/to/liqe_test_csvs \
  --batch-size 24 \
  --num-workers 8 \
  --device cuda:0 \
  --fusion-weight 1.0 \
  --output-dir results/iqag_test
