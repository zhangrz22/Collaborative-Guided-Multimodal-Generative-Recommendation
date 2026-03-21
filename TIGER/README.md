# CEMG TIGER Pipeline

## 1) Build SID data

Run in `CEMG/tokenizer`:

```bash
python3 build_sid_data.py
```

`build_sid_data.py` now uses in-file defaults (edit `DEFAULT_CONFIG` inside the script if needed).
Default output folder: `data/tiger_data/`.
Default input `item_codes.parquet`: `data/tiger_data/item_codes.parquet`.

This generates:
- `data/tiger_data/Beauty/merge.index.json`
- `data/tiger_data/Beauty/Beauty.inter.json`
- `data/tiger_data/Beauty/item_sid_map.json`
- `data/tiger_data/Beauty/Beauty.pretrain.json` (optional, with new `sid`)

## 2) 8-GPU training

Run in `CEMG/TIGER`:

```bash
chmod +x train_8gpu.sh
./train_8gpu.sh
```

Override with env vars if needed:

```bash
DATA_PATH=../data/tiger_data \
DATASET=Beauty \
NUM_GPUS=8 \
BASE_MODEL=/llm-reco-ssd-share/zhangrongzhou/Tiger/pretrained_models/t5_config \
./train_8gpu.sh
```

If `DATA_PATH` is not set, it defaults to `../data/tiger_data`.
If `BASE_MODEL` is not set, it defaults to `/llm-reco-ssd-share/zhangrongzhou/Tiger/pretrained_models/t5_config`.
Training script already evaluates on validation during training and runs final test evaluation with the best checkpoint after training ends.
