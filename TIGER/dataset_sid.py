import json
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import T5Tokenizer


class SIDDataset(Dataset):
    """
    Dataset for TIGER SID data.

    inter format:
    {
      "0": [9449, 9839, ...],
      "1": [3309, 4572, ...]
    }

    index format:
    {
      "0": ["<s_a_3>", "<s_b_189>", "<s_c_128>", "<s_d_88>"],
      "1": ["<s_a_63>", "<s_b_72>", "<s_c_218>", "<s_d_36>"]
    }
    """

    def __init__(self, inter_path, index_path, tokenizer, mode="train", max_len=20, pad_token_id=0):
        self.inter_path = inter_path
        self.index_path = index_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        with open(inter_path, "r", encoding="utf-8") as f:
            self.inters: Dict[str, List[int]] = json.load(f)
        with open(index_path, "r", encoding="utf-8") as f:
            self.indices: Dict[str, List[str]] = json.load(f)

        self.data = self._prepare_data()

    def _item_to_token_ids(self, item_id):
        sid_tokens = self.indices[str(item_id)]
        token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in sid_tokens]
        return token_ids

    def _process_history(self, history_items):
        history_token_ids = []
        for item_id in history_items:
            history_token_ids.extend(self._item_to_token_ids(item_id))

        max_tokens = self.max_len * 4
        if len(history_token_ids) > max_tokens:
            history_token_ids = history_token_ids[-max_tokens:]
        if len(history_token_ids) < max_tokens:
            history_token_ids = [self.pad_token_id] * (max_tokens - len(history_token_ids)) + history_token_ids
        return history_token_ids

    def _prepare_data(self):
        processed_data = []
        for _, item_list in self.inters.items():
            if self.mode == "train":
                items = item_list[:-2]
                for i in range(1, len(items)):
                    history_items = items[:i]
                    target_item = items[i]
                    processed_data.append(
                        {
                            "history": self._process_history(history_items),
                            "target": self._item_to_token_ids(target_item),
                            "target_item": int(target_item),
                        }
                    )
            elif self.mode == "valid":
                if len(item_list) < 3:
                    continue
                history_items = item_list[:-2]
                target_item = item_list[-2]
                processed_data.append(
                    {
                        "history": self._process_history(history_items),
                        "target": self._item_to_token_ids(target_item),
                        "target_item": int(target_item),
                    }
                )
            elif self.mode == "test":
                if len(item_list) < 2:
                    continue
                history_items = item_list[:-1]
                target_item = item_list[-1]
                processed_data.append(
                    {
                        "history": self._process_history(history_items),
                        "target": self._item_to_token_ids(target_item),
                        "target_item": int(target_item),
                    }
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_sid_tokenizer(base_model_path, index_path):
    tokenizer = T5Tokenizer.from_pretrained(base_model_path)

    with open(index_path, "r", encoding="utf-8") as f:
        indices = json.load(f)

    sid_tokens = set()
    for _, tokens in indices.items():
        sid_tokens.update(tokens)

    sid_tokens = sorted(sid_tokens)
    num_added = tokenizer.add_tokens(sid_tokens)
    print(f"Added {num_added} SID tokens to tokenizer.")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    return tokenizer
