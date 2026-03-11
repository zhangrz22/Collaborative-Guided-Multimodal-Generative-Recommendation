import copy
import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_valid(model, dataset, args):
    """Evaluate on validation set (full item evaluation)"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)

    # All items for evaluation (excluding padding)
    all_items = np.arange(1, itemnum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        # Build sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Predict scores for ALL items
        predictions = model.predict(*[np.array(l) for l in [[u], [seq], all_items]])
        predictions = predictions[0]  # Shape: (itemnum,)

        # Get rank of target item (valid[u][0])
        target_item = valid[u][0]
        target_score = predictions[target_item - 1]  # item IDs start from 1
        rank = (predictions > target_score).sum().item()  # Count items with higher score

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_test(model, dataset, args):
    """Evaluate on test set (full item evaluation)"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)

    # All items for evaluation (excluding padding)
    all_items = np.arange(1, itemnum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # Build sequence (include validation item)
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Predict scores for ALL items
        predictions = model.predict(*[np.array(l) for l in [[u], [seq], all_items]])
        predictions = predictions[0]  # Shape: (itemnum,)

        # Get rank of target item (test[u][0])
        target_item = test[u][0]
        target_score = predictions[target_item - 1]  # item IDs start from 1
        rank = (predictions > target_score).sum().item()  # Count items with higher score

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_full(model, dataset, args, metrics=["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]):
    """Evaluate with multiple metrics (full item evaluation for test)"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    metrics_dict = {m: 0.0 for m in metrics}
    valid_user = 0.0

    users = range(1, usernum + 1)

    # All items for evaluation (excluding padding)
    all_items = np.arange(1, itemnum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # Build sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Predict scores for ALL items
        predictions = model.predict(*[np.array(l) for l in [[u], [seq], all_items]])
        predictions = predictions[0]  # Shape: (itemnum,)

        # Get rank of target item (test[u][0])
        target_item = test[u][0]
        target_score = predictions[target_item - 1]  # item IDs start from 1
        rank = (predictions > target_score).sum().item()  # Count items with higher score

        valid_user += 1

        # Calculate metrics
        for metric in metrics:
            if metric.startswith("hit@"):
                k = int(metric.split("@")[1])
                if rank < k:
                    metrics_dict[metric] += 1
            elif metric.startswith("ndcg@"):
                k = int(metric.split("@")[1])
                if rank < k:
                    metrics_dict[metric] += 1 / np.log2(rank + 2)

    # Average
    for metric in metrics:
        metrics_dict[metric] /= valid_user

    return metrics_dict
