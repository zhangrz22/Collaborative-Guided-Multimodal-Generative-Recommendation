import argparse
import os
import time
import torch
import torch.distributed as dist
import numpy as np

from model import SASRec
from dataset import load_data, WarpSampler
from utils import set_seed, evaluate_valid, evaluate_test, evaluate_full


def train(args):
    set_seed(args.seed)

    # Setup DDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world_size > 1

    if ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.device = device
    else:
        device = torch.device(args.device)

    # Create output directory
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    dataset = load_data(args)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # Create sampler
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    if local_rank == 0:
        print(f"Number of batches per epoch: {num_batch}")

    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        num_neg=args.num_neg,
        n_workers=args.num_workers
    )

    if local_rank == 0:
        print(f"Negative samples per position: {args.num_neg}")
        if args.num_neg > 1:
            print("Using Cross Entropy loss")
        else:
            print("Using BPR loss (BCE)")

    # Create model
    model = SASRec(usernum, itemnum, args).to(device)

    # Initialize parameters
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Zero out padding embeddings
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    if local_rank == 0:
        print(model)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap with DDP
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # Loss and optimizer
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Training loop
    best_val_hr = 0.0
    best_val_ndcg = 0.0
    best_model_path = None
    T = 0.0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            optimizer.zero_grad()

            if args.num_neg == 1:
                # BPR loss (original SASRec)
                neg = neg.squeeze(-1) if len(neg.shape) == 3 else neg  # [B, T, 1] -> [B, T]
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)

                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            else:
                # Cross Entropy loss with multiple negatives
                base_model = model.module if ddp else model
                log_feats = base_model.log2feats(seq)  # [B, T, C]

                # Construct candidates: [B, T, num_neg+1]
                # First position is positive, rest are negatives
                pos_expanded = np.expand_dims(pos, axis=-1)  # [B, T] -> [B, T, 1]
                candidates = np.concatenate([pos_expanded, neg], axis=-1)  # [B, T, num_neg+1]

                # Get logits for all candidates
                logits = base_model.predict_candidates(log_feats, candidates)  # [B, T, num_neg+1]

                # Mask for valid positions (non-padding)
                indices = np.where(pos != 0)  # positions with valid items

                # Labels: 0 for all valid positions (positive is at index 0)
                # Flatten logits and create labels
                logits_valid = logits[indices]  # [num_valid, num_neg+1]
                labels = torch.zeros(logits_valid.size(0), dtype=torch.long, device=args.device)

                # Cross entropy loss
                loss = torch.nn.functional.cross_entropy(logits_valid, labels)

            # L2 regularization
            base_model = model.module if ddp else model
            for param in base_model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % args.log_step == 0 and local_rank == 0:
                print(f"Epoch {epoch} Step {step+1}/{num_batch}: loss = {loss.item():.4f}")

        avg_loss = epoch_loss / num_batch
        if local_rank == 0:
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Evaluation (only rank 0)
        if epoch % args.eval_epoch == 0 and local_rank == 0:
            # Get the actual model (unwrap DDP if needed)
            eval_model = model.module if ddp else model
            eval_model.eval()
            t1 = time.time() - t0
            T += t1

            print("Evaluating...")
            t_valid = evaluate_valid(eval_model, dataset, args)
            t_test = evaluate_test(eval_model, dataset, args)

            print(f"Epoch {epoch}, Time: {T:.1f}s")
            print(f"Valid: NDCG@10 = {t_valid[0]:.4f}, HR@10 = {t_valid[1]:.4f}")
            print(f"Test:  NDCG@10 = {t_test[0]:.4f}, HR@10 = {t_test[1]:.4f}")

            # Save best model (based on validation HR@10)
            if t_valid[1] > best_val_hr:
                best_val_hr = t_valid[1]
                best_val_ndcg = t_valid[0]

                save_path = os.path.join(
                    args.output_dir,
                    f"SASRec_epoch{epoch}_hr{t_valid[1]:.4f}.pth"
                )
                torch.save(eval_model.state_dict(), save_path)
                best_model_path = save_path
                print(f"Model saved to {save_path} (Best HR@10)")

            t0 = time.time()

    # Save final model (only rank 0)
    if local_rank == 0:
        eval_model = model.module if ddp else model
        final_path = os.path.join(args.output_dir, "SASRec_final.pth")
        torch.save(eval_model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")

        # Evaluate best model with full metrics
        print("\n" + "=" * 80)
        print("Final Evaluation on Best Model (based on validation HR@10)")
        print("=" * 80)
        print(f"Best validation HR@10: {best_val_hr:.4f}, NDCG@10: {best_val_ndcg:.4f}")

        # Load best model if it was saved
        if best_model_path is not None:
            print(f"Loading best model: {best_model_path}")
            eval_model.load_state_dict(torch.load(best_model_path))
            eval_model.eval()

            # Full metrics evaluation on test set
            metrics = ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]
            test_results = evaluate_full(eval_model, dataset, args, metrics)

            print("\nTest Set Results (Best Model based on validation HR@10):")
            print("-" * 80)
            for metric, value in test_results.items():
                print(f"{metric}: {value:.6f}")
            print("=" * 80 + "\n")
        else:
            print("No best model was saved during training.")

    sampler.close()

    if ddp:
        dist.destroy_process_group()

    if local_rank == 0:
        print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SASRec Training")

    # Data parameters
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="Data directory")

    # Model parameters
    parser.add_argument("--hidden_units", type=int, default=50, help="Hidden dimension")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of Transformer blocks")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--maxlen", type=int, default=50, help="Max sequence length")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--l2_emb", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--num_neg", type=int, default=1, help="Number of negative samples (1=BPR loss, >1=CE loss)")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--eval_epoch", type=int, default=10, help="Evaluation frequency")
    parser.add_argument("--log_step", type=int, default=100, help="Logging frequency")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of data workers")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./ckpt/Beauty", help="Output directory")

    args = parser.parse_args()

    print("=" * 80)
    print("Training SASRec")
    print("=" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("=" * 80)

    train(args)
