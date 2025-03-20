import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovlm_blocks import LayerNorm, TransformerBlock


class DecoderBlock(nn.Module):
    """Decoder Block for NanoVLM"""
    def __init__(self, n_embd, n_head, head_size, vocab_size, max_seq_length=512, dropout=0.1, n_layer=6):
        super().__init__()
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, n_embd))
        
        # Transformer layers (using causal attention)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, head_size, dropout, causal=True)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, multimodal_embed=None):
        B, T = x.size()
        
        # Get token embeddings
        token_embed = self.token_embedding(x)  # (B, T, n_embd)
        
        # Add positional embeddings
        pos_embed = self.pos_embedding[:, :T, :]
        x = token_embed + pos_embed
        x = self.dropout(x)
        
        # If multimodal embedding is provided, prepend it to the sequence
        if multimodal_embed is not None:
            x = torch.cat([multimodal_embed, x], dim=1)  # (B, T+mm_len, n_embd)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits