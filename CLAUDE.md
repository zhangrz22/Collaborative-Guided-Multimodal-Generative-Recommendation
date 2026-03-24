# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CEMG (Collaborative-Guided Multimodal Generative Recommendation) is a research project for sequential recommendation using Semantic IDs (SIDs). The core idea: convert item text descriptions into embeddings (via Qwen3-Embedding-8B), quantize them into discrete multi-layer codes (RQ-VAE or RQ-KMeans), then train generative models that predict the next item's SID given user history.

Primary dataset: Amazon Beauty. All models use **leave-last-two-out** splitting (train: items `[0..n-3]`, valid: item `[n-2]`, test: item `[n-1]`).

## Architecture

### Pipeline (run in order)

1. **Data preparation** (`data/`): `process_raw_data.py` → `extract_item_info.py` → `create_text_descriptions.py` → `truncate_sequences.py`
2. **Embedding generation** (`tokenizer/generate_embeddings.py`): Qwen3-Embedding-8B → `item_text_embeddings.parquet`
3. **Tokenization** (`tokenizer/`): Either `RQ-VAE/run_rq_vae.sh` or `RQ-KMeans/run_rq_kmeans.sh` → item codes parquet
4. **SID data build** (`tokenizer/build_sid_data.py`): Converts codes + interactions into TIGER/OneRec data format (`merge.index.json`, `Beauty.inter.json`, `item_sid_map.json`, etc.)
5. **Model training**: SASRec, TIGER, or OneRec (see commands below)

### SID Format

- **4 layers of codes**, e.g. `["<s_a_3>", "<s_b_189>", "<s_c_128>", "<s_d_88>"]`
- TIGER uses 4 independent vocab layers (a, b, c, d)
- OneRec uses 3 independent vocabs with shared c for positions 3 and 4: `position_to_vocab = (0, 1, 2, 2)`
- Default RQ config: 4 layers, 256 codes/layer (TIGER) or 8192 codes/layer (OneRec)
- Collision handling: item with highest interaction frequency becomes the canonical item for a shared SID

### Models

- **SASRec** (`SASRec/`): Traditional item-ID baseline. Transformer with causal self-attention, BPR/CE loss. Metrics: HR@k, NDCG@k with full-item ranking.
- **TIGER** (`TIGER/`): T5-based generative recommender. Wraps `T5ForConditionalGeneration`, adds SID tokens to vocabulary, uses trie-based constrained beam search. Metrics: Recall@k, NDCG@k.
- **OneRec** (`OneRec/`): Custom Transformer with cross-attention (history context → KV cache) + self-attention. Uses RMSNorm, SiLU-gated FFN, beam search with precut + top-k pruning. Metrics: Hit@k, Precision, Recall, NDCG.

### RQ-VAE (`tokenizer/RQ-VAE/`)

- `rq_vae.py`: Encoder (2-layer MLP + GELU) → mu head → Residual Vector Quantizer → Decoder. EMA codebook updates, dead code restart, K-Means init, Sinkhorn-Knopp collision refinement.
- Losses: MSE reconstruction + commitment + optional KL.

### RQ-KMeans (`tokenizer/RQ-KMeans/`)

- `res_kmeans.py`: Simpler alternative using faiss K-Means. Each layer fits on residual of previous layer.

## Commands

All training scripts run via `nohup` in background and default to 8 GPUs. All parameters are overridable via environment variables.

### Tokenization

```bash
# RQ-VAE
cd tokenizer/RQ-VAE && ./run_rq_vae.sh

# RQ-KMeans
cd tokenizer/RQ-KMeans && ./run_rq_kmeans.sh

# Build SID data (after tokenization)
cd tokenizer && python3 build_sid_data.py
```

### SASRec (baseline)

```bash
cd SASRec
python prepare_data.py          # convert interaction sequences to SASRec format
./run_train.sh                  # 8-GPU torchrun training
```

### TIGER

```bash
cd TIGER
./train_8gpu.sh                 # 8-GPU torchrun training + validation + test

# Override defaults:
DATA_PATH=../data/tiger_data DATASET=Beauty NUM_GPUS=4 ./train_8gpu.sh

# Standalone test:
python3 test_sid.py --checkpoint <path> ...
```

### OneRec

```bash
cd OneRec
./run.sh                        # 8-GPU DDP training

# Override defaults:
DATASET=Beauty DATA_PATH=../data/tiger_data BATCH_SIZE=64 ./run.sh

# Test/inference:
CHECKPOINT=./checkpoints/Beauty/<timestamp>/best.pt ./run_test.sh
```

## Key Dependencies (no requirements.txt)

Python 3, PyTorch (DDP, torchrun, AMP/bfloat16), HuggingFace Transformers (T5), NumPy, Pandas, faiss (RQ-KMeans), flash_attn (embedding generation), torchmetrics (OneRec), tensorboard (OneRec), tqdm.

## Important Details

- **No test suite exists.** There are no unit or integration tests.
- All shell scripts use `nohup` and run in background; check logs with `tail -f <log_file>`.
- Shared data directory: `data/tiger_data/` is used by both TIGER and OneRec.
- TIGER requires a T5 config directory (`BASE_MODEL` env var) for model architecture initialization.
- OneRec supports both JSON (LC-Rec style, default) and Parquet data formats via `--data_format`.
- Multi-GPU: SASRec and TIGER use `torchrun`; OneRec uses `mp.spawn`.
