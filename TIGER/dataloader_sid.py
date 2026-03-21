import torch
from torch.utils.data import DataLoader


def sid_collate_fn(batch, pad_token_id=0):
    histories = [item["history"] for item in batch]
    targets = [item["target"] for item in batch]

    history_tensor = torch.tensor(histories, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    attention_mask = (history_tensor != pad_token_id).long()

    return {
        "history": history_tensor,
        "target": target_tensor,
        "attention_mask": attention_mask,
    }


def build_sid_dataloader(dataset, batch_size, shuffle, num_workers, pad_token_id, sampler=None):
    # shuffle must be False when sampler is specified
    effective_shuffle = shuffle if sampler is None else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: sid_collate_fn(batch, pad_token_id=pad_token_id),
        pin_memory=True,
        drop_last=False,
    )

