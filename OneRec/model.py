from typing import List, Union, Tuple
from pprint import pformat
import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    device: str = 'cuda'
    target_type_num: int = 1
    beam_size: int = 10
    precut_num: int = 1024
    max_his_len: int = 50
    semantic_token_num: int = 4  # a, b, c, c (4个位置)
    num_vocab_layers: int = 3    # a, b, c (3个独立词表)
    position_to_vocab: tuple = (0, 1, 2, 2)  # 位置到词表的映射: [a, b, c, c]
    sid_embedding_dim: int = 1024
    d_model: int = 1024
    ffw_size: int = None
    n_layers: int = 12
    n_heads: int = 16
    vocab_size: int = 8192
    dropout: float = 0.
    kv_layers: int = 1
    kv_head_group_nums: int = None
    kv_split: bool = False
    head_dim: int = None
    def __post_init__(self):
        if self.ffw_size is None:
            self.ffw_size = int(self.d_model * 4)
        if self.kv_head_group_nums is None:
            self.kv_head_group_nums = self.n_heads
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
            self.head_dim = self.d_model // self.n_heads
        assert self.n_heads % self.kv_head_group_nums == 0, "n_heads must be divisible by kv_head_group_nums"
        assert self.n_layers % self.kv_layers == 0, "n_layers must be divisible by kv_layers"
        self.kv_head_group_size = self.n_heads // self.kv_head_group_nums
        self.kv_share_size = self.n_layers // self.kv_layers
        self.kv_split_coe = 2 if self.kv_split else 1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) * self.weight
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CrossAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wq = nn.Linear(self.config.d_model, self.config.head_dim * self.config.n_heads, bias=False)
        self.wo = nn.Linear(self.config.head_dim * self.config.n_heads, self.config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(self.config.dropout)

    def forward(self, q, k, v):
        q = self.wq(q)
        q_bs = q.size(0)
        q_seq_len = q.size(1)

        beam_size = q_bs // k.size(0)
        q = q.view(k.size(0), beam_size * q_seq_len * self.config.kv_head_group_size, self.config.kv_head_group_nums, -1).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                               dropout_p=self.config.dropout if self.training else 0.,
                                                               is_causal=False)
        out = out.transpose(1, 2).contiguous().view(q_bs, q_seq_len, -1)

        out = self.wo(out)
        out = self.resid_dropout(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wq = nn.Linear(self.config.d_model, self.config.head_dim * self.config.n_heads, bias=False)
        self.wk = nn.Linear(self.config.d_model, self.config.head_dim * self.config.n_heads, bias=False)
        self.wv = nn.Linear(self.config.d_model, self.config.head_dim * self.config.n_heads, bias=False)
        self.wo = nn.Linear(self.config.head_dim * self.config.n_heads, self.config.d_model, bias=False)

        self.resid_dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, kv_cache=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(q.size(0), q.size(1), self.config.n_heads, -1).transpose(1, 2) # [b, h, sq, d]
        k = k.view(k.size(0), k.size(1), self.config.n_heads, -1).transpose(1, 2) # [b, h, sq, d]
        v = v.view(v.size(0), v.size(1), self.config.n_heads, -1).transpose(1, 2) # [b, h, sq, d]

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = [k, v]

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                               dropout_p=self.config.dropout if self.training else 0.,
                                                               is_causal=self.training)
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1)

        out = self.wo(out)
        out = self.resid_dropout(out)
        return out, new_kv_cache


class OneRecBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.cross_attn = CrossAttention(config)
        self.self_attn = SelfAttention(config)
        self.self_ffn = FeedForward(config.d_model, config.ffw_size, config.dropout)
        self.cross_attn_norm = RMSNorm(config.d_model, eps=1e-5)
        self.self_att_norm = RMSNorm(config.d_model, eps=1e-5)
        self.self_ffn_norm = RMSNorm(config.d_model, eps=1e-5)

    def forward(self, x, k, v, self_kv_cache=None):
        x = x + self.cross_attn(self.cross_attn_norm(x), k, v)

        h, self_kv_cache = self.self_attn(self.self_att_norm(x), self_kv_cache)
        h = x + h
        h = h + self.self_ffn(self.self_ffn_norm(h))
        return h, self_kv_cache


class ContextProcessor(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        fc_dim = self.config.head_dim * self.config.kv_head_group_nums * self.config.kv_layers * self.config.kv_split_coe

        self.sid_fc = nn.Linear(config.sid_embedding_dim, fc_dim, bias=False)
        self.kv_norms = nn.ModuleList(
            [RMSNorm(self.config.head_dim * self.config.kv_head_group_nums, eps=1e-5)
             for _ in range(self.config.kv_layers * self.config.kv_split_coe)])

        # ★ 只创建 num_vocab_layers 个独立词表 (a, b, c)
        self.sid_embs = torch.nn.ModuleList([
            torch.nn.Embedding(config.vocab_size + 1, config.sid_embedding_dim, padding_idx=0)
            for _ in range(self.config.num_vocab_layers)  # 只创建 3 个
        ])

        self.begin_end_embs = torch.nn.Embedding(2, fc_dim)
        self.pos_embs = torch.nn.Embedding(config.max_his_len * self.config.semantic_token_num + 1, fc_dim, padding_idx=0)
        self.type_embs = torch.nn.Embedding(config.target_type_num, fc_dim)

    def forward(self, his_sids, his_pid_types):
        """
        Args:
            his_sids: [bs, max_his_len * sid_len] - sid序列
            his_pid_types: [bs, target_type_num, max_his_len] - bool tensor
                         - 每个类型的bool标记（一个pid可以有多个类型）
        """
        bs = his_sids.shape[0]
        pos = torch.arange(1, self.config.max_his_len * self.config.semantic_token_num + 1,
                           device=his_sids.device).unsqueeze(0).tile(bs, 1)
        pos[his_sids==-1] = 0
        pos_embs = self.pos_embs(pos) #  [bs, max_his_len * sid_len, fc_dim]

        begin_end_embs = torch.tensor([0, 1], device=his_sids.device, dtype=torch.long)
        begin_end_embs = self.begin_end_embs(begin_end_embs)  # shape: [2, fc_dim]
        begin_end_embs = begin_end_embs.unsqueeze(1).unsqueeze(1).tile(1, bs, self.config.max_his_len, 1)
        # shape: [2, bs, max_his_len, fc_dim]

        his_sids = his_sids + 1
        his_sids = his_sids.view(bs, self.config.max_his_len, self.config.semantic_token_num)

        # ★ 使用 position_to_vocab 映射来查表
        sid_embs = torch.stack([
            self.sid_embs[self.config.position_to_vocab[i]](his_sids[:, :, i])
            for i in range(his_sids.shape[-1])
        ], dim=2)
        # position_to_vocab = (0, 1, 2, 2)
        # i=0 → sid_embs[0] (a)
        # i=1 → sid_embs[1] (b)
        # i=2 → sid_embs[2] (c)
        # i=3 → sid_embs[2] (c) ← 共享

        sid_embs = self.sid_fc(sid_embs) #  [bs, max_his_len, sid_len, fc_dim]
        sid_embs = sid_embs + pos_embs.view(sid_embs.shape)
        sid_embs = torch.cat([begin_end_embs[0].unsqueeze(2), sid_embs, begin_end_embs[1].unsqueeze(2)], dim=2)
        #  [bs, max_his_len, sid_len + 2, fc_dim]

        # 处理type embeddings - 支持多标签
        type_embs = torch.zeros(bs, self.config.max_his_len, sid_embs.size(-1), device=his_sids.device, dtype=self.type_embs.weight.dtype)
        #  [bs, max_his_len, fc_dim]
        for type_idx in range(self.config.target_type_num):
            type_mask = his_pid_types[:, type_idx, :]  # [bs, max_his_len]
            # 对于标记为该类型的位置，添加对应的type embedding
            type_emb = self.type_embs(torch.full([bs, self.config.max_his_len], type_idx, device=his_sids.device, dtype=torch.long))  # type_idx对应embedding索引
            type_embs = type_embs + type_emb * type_mask.unsqueeze(-1).to(type_emb.dtype)

        kv = sid_embs + type_embs.unsqueeze(-2)
        kv = kv.view(kv.size(0), kv.size(1) * kv.size(2), kv.size(3))
        kv = kv.chunk(self.config.kv_layers * self.config.kv_split_coe, dim=-1)

        context_list = []
        for i in range(self.config.kv_layers):
            k_pos = i * self.config.kv_split_coe
            k = self.kv_norms[k_pos](kv[k_pos])
            k = k.view(k.size(0), k.size(1), self.config.kv_head_group_nums, -1).transpose(1, 2)
            if self.config.kv_split:
                v = self.kv_norms[k_pos + 1](kv[k_pos + 1])
                v = v.view(v.size(0), v.size(1), self.config.kv_head_group_nums, -1).transpose(1, 2)
            else:
                v = k
            context_list.append((k, v))
        return context_list


class OneRecV2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        print("=" * 80)
        print(f"model config: {pformat(config)}")
        print("=" * 80)

        self.context_processor = ContextProcessor(config)
        self.layers = nn.ModuleList([OneRecBlock(config) for _ in range(config.n_layers)])

        self.out_norm = RMSNorm(self.config.d_model, eps=1e-5)

        # ★ Target 输入: 为每个位置创建独立的 Embedding (除了 BOS)
        # 但使用 position_to_vocab 映射来复用词表
        self.tok_embeddings = torch.nn.ModuleList([
            nn.Embedding(self.config.vocab_size, self.config.d_model)
            for _ in range(self.config.num_vocab_layers)  # 只创建 3 个 (a, b, c)
        ])

        self.bos_embedding = nn.Embedding(self.config.target_type_num, self.config.d_model)
        self.pos_embedding = nn.Embedding(self.config.semantic_token_num, self.config.d_model)

        # ★ Target 输出: 为每个位置创建独立的输出层
        self.output = torch.nn.ModuleList([
            nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            for _ in range(self.config.num_vocab_layers)  # 只创建 3 个
        ])

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        self.to(config.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, his_sids, his_pid_types, target_sids, target_type):
        bs = his_sids.shape[0]

        context_list = self.context_processor(his_sids, his_pid_types)
        bos_embedding = self.bos_embedding(target_type).unsqueeze(1)

        # ★ 使用 position_to_vocab 映射来查表 (跳过 BOS 位置)
        token_embeds = [
            self.tok_embeddings[self.config.position_to_vocab[i]](target_sids[:, i]).unsqueeze(1)
            for i in range(self.config.semantic_token_num - 1)
        ]
        # i=0 → position_to_vocab[0]=0 → tok_embeddings[0] (a)
        # i=1 → position_to_vocab[1]=1 → tok_embeddings[1] (b)
        # i=2 → position_to_vocab[2]=2 → tok_embeddings[2] (c)

        h = torch.cat([bos_embedding] + token_embeds, dim=1)
        h = h + self.pos_embedding.weight[None, :, :]

        for layer_idx, layer in enumerate(self.layers):
            k, v = context_list[layer_idx // self.config.kv_share_size]
            h, _ = layer(h, k, v)

        h = self.out_norm(h)

        # ★ 使用 position_to_vocab 映射来预测
        h = [
            self.output[self.config.position_to_vocab[i]](h[:, i, :]).unsqueeze(1)
            for i in range(self.config.semantic_token_num)
        ]
        # i=0 → output[0] (预测 a)
        # i=1 → output[1] (预测 b)
        # i=2 → output[2] (预测 c)
        # i=3 → output[2] (预测 c) ← 共享输出层

        h = torch.cat(h, dim=1)  # [bs, semantic_token_num, vocab_size]
        loss_per_token = F.cross_entropy(
            h.view(-1, self.config.vocab_size),
            target_sids.view(-1),
            reduction='none'
        ).view(bs, self.config.semantic_token_num)

        pred_token_id = torch.argmax(h, dim=-1)  # [bs, semantic_token_num]
        acc = (pred_token_id == target_sids)  # [bs, semantic_token_num]
        position_acc = acc.float().mean(dim=0)
        all_correct = acc.all(dim=1).float().mean()
        loss_per_token = loss_per_token.mean(dim=0)

        return {
            "ntp_loss": loss_per_token.mean(),
            "all_correct": all_correct,
            **{f"position_acc/position_acc_{p}": position_acc[p] for p in range(self.config.semantic_token_num)},
            **{f"position_loss/position_loss_{p}": loss_per_token[p] for p in range(self.config.semantic_token_num)},
        }

    def _build_trie_masks(self, trie, vocab_size, num_positions):
        """
        将 trie 预编译为 prefix → allowed_mask 的查找表.
        每个 prefix (tuple of ints) 映射到一个 bool tensor [vocab_size].
        position 0 的 key 是 ()（空 tuple）.

        Returns:
            dict: {prefix_tuple: torch.BoolTensor of shape [vocab_size]}
        """
        prefix_masks = {}

        def _walk(node, prefix):
            keys = list(node.keys())
            if keys:
                mask = torch.zeros(vocab_size, dtype=torch.bool)
                for k in keys:
                    mask[k] = True
                prefix_masks[prefix] = mask
                for k in keys:
                    if isinstance(node[k], dict) and len(node[k]) > 0:
                        _walk(node[k], prefix + (k,))

        _walk(trie, ())
        return prefix_masks

    def generate(self, his_sids, his_pid_types, target_type, trie=None):
        bs = his_sids.shape[0]

        context_list = self.context_processor(his_sids, his_pid_types)
        use_trie = trie is not None

        # 预编译 trie 为 prefix → mask 查找表
        if use_trie:
            prefix_masks = self._build_trie_masks(trie, self.config.vocab_size, self.config.semantic_token_num)

        all_self_kv_cache = [None] * len(self.layers)
        for i in range(self.config.semantic_token_num):
            if i == 0:
                h = self.bos_embedding(target_type).unsqueeze(1).unsqueeze(1)
            else:
                # ★ 使用 position_to_vocab 映射来查表
                vocab_idx = self.config.position_to_vocab[i - 1]
                h = self.tok_embeddings[vocab_idx](semantic_id)
            # h [bs, beam_size, 1, d_model]  context [bs, kv_len, d_model]
            h = h + self.pos_embedding(torch.tensor([[[i]]], dtype=torch.long, device=h.device))
            h = h.view(h.size(0) * h.size(1), 1, h.size(3))  # h [bs * beam_size, 1, d_model]
            for layer_idx, layer in enumerate(self.layers):
                k, v = context_list[layer_idx // self.config.kv_share_size]
                h, self_kv_cache = layer(h, k, v, all_self_kv_cache[layer_idx])
                all_self_kv_cache[layer_idx] = self_kv_cache
            h = self.out_norm(h)
            # ★ 使用 position_to_vocab 映射来预测
            vocab_idx = self.config.position_to_vocab[i]
            h = self.output[vocab_idx](h)[:, -1, :]
            token_probs = F.log_softmax(h, dim=-1)

            if use_trie:
                # === Trie 约束模式：跳过 precut，直接在全 vocab 上做 trie mask + beam 选择 ===
                num_beams = 1 if i == 0 else self.config.beam_size
                n = bs * num_beams  # total number of beams

                # 构建 mask [n, vocab_size]
                mask = torch.zeros(n, self.config.vocab_size, dtype=torch.bool, device=h.device)
                if i == 0:
                    # 所有 beam 共享同一个 root mask
                    root_mask = prefix_masks.get((), None)
                    if root_mask is not None:
                        mask[:] = root_mask.to(h.device)
                    else:
                        mask[:] = True
                else:
                    # path_tokens: [bs, beam_size, i] — 用 CPU 查表，批量写入 GPU mask
                    pt_cpu = path_tokens.cpu()
                    for b in range(bs):
                        for beam in range(self.config.beam_size):
                            prefix = tuple(pt_cpu[b, beam].tolist())
                            m = prefix_masks.get(prefix, None)
                            if m is not None:
                                mask[b * self.config.beam_size + beam] = m.to(h.device)
                            else:
                                mask[b * self.config.beam_size + beam] = True

                token_probs = token_probs.masked_fill(~mask, float('-inf'))

                if i == 0:
                    topk_probs, topk_tokens = torch.topk(token_probs, k=self.config.beam_size)  # [bs, beam_size]
                    topk_probs = topk_probs.view(bs, self.config.beam_size, 1)
                    topk_tokens = topk_tokens.view(bs, self.config.beam_size, 1)
                    path_tokens = topk_tokens
                    path_probs = topk_probs
                    semantic_id = path_tokens
                    beam_indices = torch.zeros(bs, self.config.beam_size, dtype=torch.long, device=h.device)
                else:
                    # token_probs: [bs * beam_size, vocab_size]
                    accu_probs = path_probs + token_probs.view(bs, self.config.beam_size, self.config.vocab_size)
                    # [bs, beam_size, vocab_size]
                    accu_probs = accu_probs.view(bs, self.config.beam_size * self.config.vocab_size)
                    topk_probs, topk_indices = accu_probs.topk(k=self.config.beam_size)  # [bs, beam_size]
                    topk_tokens = topk_indices % self.config.vocab_size  # [bs, beam_size]
                    beam_indices = topk_indices // self.config.vocab_size  # [bs, beam_size]
                    path_tokens = torch.gather(path_tokens, dim=1, index=beam_indices.unsqueeze(-1).expand(-1, -1, i))
                    path_tokens = torch.cat([path_tokens, topk_tokens.unsqueeze(-1)], dim=2)  # [bs, beam_size, i+1]
                    semantic_id = path_tokens[:, :, -1:]  # [bs, beam_size, 1]
                    path_probs = topk_probs.view(bs, self.config.beam_size, 1)
            else:
                # === 原始逻辑：precut topk ===
                token_probs, token_indices = token_probs.topk(k=self.config.precut_num, dim=-1)  # [bs * beam_size, precut_num]

                if i == 0:
                    topk_probs, topk_indices = torch.topk(token_probs, k=self.config.beam_size)  # [bs * 1, beam_size]
                    topk_tokens = torch.gather(token_indices, 1, topk_indices)  # [bs * 1, beam_size]
                    topk_probs = topk_probs.view(bs, self.config.beam_size, 1)
                    topk_tokens = topk_tokens.view(bs, self.config.beam_size, 1)
                    path_tokens = topk_tokens
                    path_probs = topk_probs
                    semantic_id = path_tokens
                else:
                    accu_probs = path_probs + token_probs.view(bs, self.config.beam_size, self.config.precut_num)
                    # [bs, beam_size, 1] + [bs, beam_size, precut_num]
                    accu_probs = accu_probs.view(bs, self.config.beam_size * self.config.precut_num)
                    topk_probs, topk_indices = accu_probs.topk(k=self.config.beam_size)  # [bs, beam_size]
                    topk_tokens = torch.gather(token_indices.view(bs, self.config.beam_size * self.config.precut_num), 1, topk_indices)  # [bs, beam_size]
                    beam_indices = topk_indices // self.config.precut_num  # [bs, beam_size]
                    path_tokens = torch.gather(path_tokens, dim=1, index=beam_indices.unsqueeze(-1).expand(-1, -1, i))
                    path_tokens = torch.cat([path_tokens, topk_tokens.unsqueeze(-1)], axis=2)  # [bs, beam_size, i+1]
                    semantic_id = path_tokens[:, :, -1:]  # [bs, beam_size, 1]
                    path_probs = topk_probs.view(bs, self.config.beam_size, 1)

                beam_indices = topk_indices // self.config.precut_num  # [bs, beam_size]

            batch_idx = torch.arange(bs, device=h.device).unsqueeze(1).expand(bs, self.config.beam_size)
            for layer_idx in range(len(self.layers)):
                k_cache, v_cache = all_self_kv_cache[layer_idx]  # [bs * beam_size, n_head, i+1, head_dim]
                k_cache = (
                    k_cache.view(bs, -1, self.config.n_heads, i + 1, self.config.head_dim)[batch_idx, beam_indices]
                    .view(bs * self.config.beam_size, self.config.n_heads, i + 1, self.config.head_dim))
                v_cache = (
                    v_cache.view(bs, -1, self.config.n_heads, i + 1, self.config.head_dim)[batch_idx, beam_indices]
                    .view(bs * self.config.beam_size, self.config.n_heads, i + 1, self.config.head_dim))
                all_self_kv_cache[layer_idx] = [k_cache, v_cache]

        return {
            "tokens": path_tokens,
            "scores": path_probs
        }
