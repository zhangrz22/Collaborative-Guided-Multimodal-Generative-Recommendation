import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation Model"""
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """Convert item sequences to features using Transformer"""
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Position encoding
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Causal attention mask
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # Transformer blocks
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            x = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
            seqs = seqs + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training with BPR loss"""
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """Predict scores for given items (for inference)"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use last position

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def predict_candidates(self, log_feats, candidates):
        """
        Predict scores for candidate items given hidden states
        Args:
            log_feats: [B, T, C] hidden states from log2feats
            candidates: [B, T, N] candidate item IDs (positive + negatives)
        Returns:
            logits: [B, T, N] scores for each candidate
        """
        # log_feats: [B, T, C]
        # candidates: [B, T, N] where N = 1 + num_neg
        B, T, N = candidates.shape
        C = log_feats.shape[-1]

        # Get embeddings for all candidates
        # candidates: [B, T, N] -> [B*T*N]
        candidates_flat = candidates.reshape(-1)
        candidate_embs = self.item_emb(torch.LongTensor(candidates_flat).to(self.dev))
        # candidate_embs: [B*T*N, C] -> [B, T, N, C]
        candidate_embs = candidate_embs.reshape(B, T, N, C)

        # Compute logits: dot product between hidden states and candidate embeddings
        # log_feats: [B, T, C] -> [B, T, 1, C]
        # candidate_embs: [B, T, N, C]
        # logits: [B, T, N]
        logits = (log_feats.unsqueeze(2) * candidate_embs).sum(dim=-1)

        return logits
