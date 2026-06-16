# QAFD / IQAG Clean Test Project

This version is organized around **`test_IQAG.py` as the only evaluation
entry point**. Machine-specific paths, inactive experiment commands, duplicate
scripts, cached files, and unrelated model families have been removed.

## Project structure

```text
QAFD_clean/
├── test_IQAG.py                  # primary evaluation entry
├── configs/
│   ├── datasets.py               # default benchmark generator lists
│   └── datasets.example.json     # dataset-root configuration template
├── data/
│   └── test_dataset.py           # path discovery, transforms, IQA prompts
├── models/
│   ├── iqag.py                   # IQAG backbone and inference wrapper
│   ├── clip_gemdwt/              # required custom CLIP implementation
│   ├── ffcresnet/                # required frequency module
│   ├── loralib/                  # optional LoRA support
│   └── dctlayer.py               # required model dependency
├── utils/
│   ├── metrics.py
│   └── runtime.py
├── scripts/test.sh               # one portable example command
├── requirements.txt
└── REMOVED_FILES.md
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare paths

Copy the dataset template and replace every placeholder with the actual path:

```bash
cp configs/datasets.example.json configs/datasets.json
```

Place the CLIP checkpoint at `pretrained/ViT-L-14.pt`, or pass a valid local
path through `--clip-checkpoint`.

## 3. Run evaluation

```bash
python test_IQAG.py \
  --dataset-config configs/datasets.json \
  --datasets cnnspot genimage \
  --checkpoint checkpoints/IQAG_best.pth \
  --clip-checkpoint pretrained/ViT-L-14.pt \
  --iqa-csv-dir /path/to/liqe_test_csvs \
  --batch-size 24 \
  --device cuda:0 \
  --output-dir results/iqag_test
```

To evaluate selected generators from one dataset:

```bash
python test_IQAG.py \
  --dataset-config configs/datasets.json \
  --datasets genimage \
  --generators ADM Midjourney \
  --checkpoint checkpoints/IQAG_best.pth \
  --clip-checkpoint pretrained/ViT-L-14.pt
```

## IQA prompt CSV

The recommended CSV layout is:

```csv
image_path,label,iqa_prompt
/path/to/image.png,1,"a photo of a landscape with blur artifacts, which has a perceptual quality of 3.2"
```

When `--iqa-csv-dir DIR` is used, the script searches for:

```text
DIR/<dataset>/<generator>.csv
```

If no CSV is found, the test uses a label-independent neutral IQA prompt. It
never generates the IQA prompt from the ground-truth real/fake label.

## Outputs

The output directory contains:

- `predictions/<dataset>/<generator>.csv`: per-image probabilities;
- `summary.csv`: per-generator metrics;
- `summary.json`: complete configuration and metric summary.

For every generator, the script reports:

- classification-head scores;
- prompt-similarity scores;
- fused scores: `classification + fusion_weight × similarity`;
- ACC, AUC, AP, real-class accuracy, and fake-class accuracy.


