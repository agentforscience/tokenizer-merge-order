"""Small GPT-style decoder used for every tokenizer condition (identical architecture, vocab = 8192 + 1 separator)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, attn_mask=None, causal=True):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal and attn_mask is None)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d=512, n_layer=6, n_head=8, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def hidden(self, idx, attn_mask=None):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, attn_mask=attn_mask)
        return self.ln_f(x)

    def forward(self, idx):
        return self.hidden(idx) @ self.wte.weight.T  # tied output embedding
