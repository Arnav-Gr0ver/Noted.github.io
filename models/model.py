import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovlm_encoder import VisualEncoder
from nanovlm_connector import VisualTextualConnector
from nanovlm_decoder import DecoderBlock


class NanoVLM(nn.Module):
    """NanoVLM: A compact Vision-Language Model"""
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3,
                 vocab_size=50304,  # Using standard GPT-2 vocabulary size
                 n_blks=6,  # Visual encoder blocks
                 n_layer=6,  # Decoder layers
                 n_head=12, 
                 head_size=64,
                 n_embd=512,  # Text embedding dimension
                 img_embd_dim=768,  # Visual embedding dimension
                 max_seq_length=512,
                 dropout=0.1):
        super().__init__()
        
        # Visual Encoder
        self.visual_encoder = VisualEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            img_embd_dim=img_embd_dim,
            n_blks=n_blks,
            n_head=n_head,
            head_size=head_size,
            dropout=dropout
        )
        
        # Visual-Textual Connector
        self.connector = VisualTextualConnector(
            img_embd_dim=img_embd_dim,
            txt_embd_dim=n_embd,
            dropout=dropout
        )
        
        # Decoder Block
        self.decoder = DecoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            head_size=head_size,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            dropout=dropout,
            n_layer=n_layer
        )

    def forward(self, img, text_tokens):
        """
        Forward pass for NanoVLM
        
        Args:
            img: Image tensor of shape (B, C, H, W)
            text_tokens: Text token indices of shape (B, T)
            
        Returns:
            logits: Output logits for next token prediction
        """
        # Process image through visual encoder
        cls_embed, _ = self.visual_encoder(img)  # (B, img_embd_dim)
        
        # Get text embeddings
        text_embeds = self.decoder.token_embedding(text_tokens)  # (B, T, n_embd)
        
        # Connect visual and textual modalities
        multimodal_embed = self.connector(cls_embed, text_embeds)  # (B, T+1, n_embd)
        
        # Generate text through decoder
        logits = self.decoder(text_tokens, multimodal_embed)  # (B, T+mm_len, vocab_size)
        
        return logits
    
    def generate(self, img, text_tokens, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text based on image and optional prompt
        
        Args:
            img: Image tensor of shape (B, C, H, W)
            text_tokens: Text token indices of shape (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: If specified, restricts sampling to top-k most likely tokens
            
        Returns:
            generated_tokens: Complete sequence including prompt and generated tokens
        """
        B, T = text_tokens.size()
        
        # Process image through visual encoder
        cls_embed, _ = self.visual_encoder(img)  # (B, img_embd_dim)
        
        # Start with given tokens (prompt)
        generated_tokens = text_tokens.clone()
        
        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # Get current sequence
            curr_tokens = generated_tokens
            
            # Get text embeddings
            text_embeds = self.decoder.token_embedding(curr_tokens)  # (B, T, n_embd)
            
            # Connect visual and textual modalities
            multimodal_embed = self.connector(cls_embed, text_embeds)  # (B, T+1, n_embd)
            
            # Forward pass through decoder
            logits = self.decoder(curr_tokens, multimodal_embed)  # (B, T+mm_len, vocab_size)
            
            # Focus on the last token's logits
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append next token to sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
        return generated_tokens