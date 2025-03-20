import torch
import torch.nn as nn


class VisualTextualConnector(nn.Module):
    """Visual-Textual Connector for NanoVLM"""
    def __init__(self, img_embd_dim=768, txt_embd_dim=512, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(img_embd_dim, txt_embd_dim),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_embed, text_embed=None):
        # Project visual embedding to text embedding space
        visual_proj = self.projector(visual_embed)  # (B, txt_embd_dim)
        
        # If text embedding is provided, concatenate with visual projection
        if text_embed is not None:
            # visual_proj: (B, txt_embd_dim)
            # text_embed: (B, L, txt_embd_dim)
            B, L, D = text_embed.shape
            visual_proj = visual_proj.unsqueeze(1)  # (B, 1, txt_embd_dim)
            
            # Concatenate along sequence dimension
            multimodal_embed = torch.cat([visual_proj, text_embed], dim=1)  # (B, L+1, txt_embd_dim)
            multimodal_embed = self.dropout(multimodal_embed)
            
            return multimodal_embed
            
        return visual_proj