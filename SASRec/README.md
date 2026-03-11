# SASRec Implementation

## Setup

1. Prepare data:
```bash
python prepare_data.py
```

2. Train model:
```bash
chmod +x run_train.sh
./run_train.sh
```

## Files
- `prepare_data.py`: Convert interaction sequences to SASRec format
- `train.py`: Training script
- `model.py`: SASRec model
- `dataset.py`: Data loader
- `utils.py`: Evaluation functions
- `run_train.sh`: Training configuration

## Output
Models saved to `./ckpt/Beauty/`
