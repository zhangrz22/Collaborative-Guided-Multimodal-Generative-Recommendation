# TIGER Data Folder

This folder is dedicated to TIGER training/evaluation artifacts to keep `data/` clean.

Expected/Generated files:
- `Beauty/item_codes.parquet` (input for SID conversion)
- `Beauty/item_sid_map.json`
- `Beauty/merge.index.json`
- `Beauty/Beauty.inter.json`
- `Beauty/Beauty.pretrain.json` (optional)

Build command:

```bash
cd tokenizer
python3 build_sid_data.py
```

`build_sid_data.py` reads/writes this folder by default.
