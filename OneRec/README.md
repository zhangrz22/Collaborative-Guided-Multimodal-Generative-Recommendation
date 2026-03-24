# OneRecV2 (CEMG)

This folder is adapted from `Baseline/OneRec` and configured to share TIGER data.

## Shared data

Default data root:

- `../data/tiger_data`

Expected files:

- `../data/tiger_data/Beauty/Beauty.inter.json`
- `../data/tiger_data/Beauty/merge.index.json`

The dataset loader now supports index path priority:
1. `{data_path}/{dataset}/merge.index.json` (preferred)
2. `{data_path}/merge/merge.index.json` (legacy fallback)

## Train (8 GPUs by default)

```bash
cd OneRec
chmod +x run.sh
./run.sh
```

Optional overrides:

```bash
DATASET=Beauty DATA_PATH=../data/tiger_data CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run.sh
```

## Test

```bash
cd OneRec
chmod +x run_test.sh
CHECKPOINT=./checkpoints/Beauty/<timestamp>/best.pt ./run_test.sh
```

