import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """Layer Normalization module"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    """Multi-head Attention module"""
    def __init__(self, n_embd, n_head, head_size, dropout=0.1, causal=False):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = head_size
        self.causal = causal
        
        # Key, Query, Value projections
        self.key = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.query = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.value = nn.Linear(n_embd, n_head * head_size, bias=False)
        
        # Output projection
        self.proj = nn.Linear(n_head * head_size, n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask for decoder
        if self.causal:
            self.register_buffer(
                "mask", 
                torch.triu(torch.ones(1, 1, 512, 512) * float('-inf'), diagonal=1)
            )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        
        # Apply causal mask if needed (for decoder)
        if self.causal:
            att = att + self.mask[:, :, :T, :T]
            
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        
        # Reshape and project back to original dimension
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)  # (B, T, C)
        y = self.resid_dropout(self.proj(y))
        
        return y


class MLP(nn.Module):
    """Multi-Layer Perceptron module"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block used in both encoder and decoder"""
    def __init__(self, n_embd, n_head, head_size, dropout=0.1, causal=False):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, dropout, causal)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        # Pre-normalization for attention
        x = x + self.attn(self.ln1(x))
        # Pre-normalization for MLP
        x = x + self.mlp(self.ln2(x))
        return x