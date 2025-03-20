import torch
import torch.nn as nn
from nanovlm_blocks import LayerNorm, TransformerBlock


class PatchEmbedding(nn.Module):
    """Image patch embedding module for the visual encoder"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, img_embd_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Two convolutional layers as described
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.ln1 = LayerNorm(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.ln2 = LayerNorm(64)
        
        # Final projection to embedding dimension
        patch_dim = (patch_size // 2) ** 2 * 64  # Accounting for stride in conv1
        self.proj = nn.Linear(patch_dim, img_embd_dim)

    def forward(self, x):
        # Input: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Apply first conv layer
        x = self.conv1(x)  # (B, 32, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, 32)
        x = self.ln1(x)
        x = self.relu(x)
        x = x.permute(0, 3, 1, 2)  # (B, 32, H/2, W/2)
        
        # Apply second conv layer
        x = self.conv2(x)  # (B, 64, H/2, W/2)
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, 64)
        x = self.ln2(x)
        x = self.relu(x)
        
        # Reshape to patches
        x = x.reshape(B, -1, (self.patch_size // 2) ** 2 * 64)  # (B, N, patch_dim)
        
        # Project to embedding dimension
        x = self.proj(x)  # (B, N, img_embd_dim)
        
        return x


class VisualEncoder(nn.Module):
    """Visual Encoder for NanoVLM"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, img_embd_dim=768, 
                n_blks=6, n_head=12, head_size=64, dropout=0.1):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, img_embd_dim)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, img_embd_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, img_embd_dim))
        
        # Normalization and dropout
        self.norm = LayerNorm(img_embd_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(img_embd_dim, n_head, head_size, dropout, causal=False)
            for _ in range(n_blks)
        ])

    def forward(self, x):
        # Input: (B, C, H, W)
        B = x.shape[0]
        
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, N, img_embd_dim)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, img_embd_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Extract CLS token for the final representation
        cls_output = x[:, 0]  # (B, img_embd_dim)
        
        return cls_output, x