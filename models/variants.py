from nanovlm_model import NanoVLM

# NanoVLM model variants as mentioned in the paper
def nano_vlm_mini(vocab_size=50304):
    return NanoVLM(
        n_blks=4,         # Visual encoder blocks
        n_layer=4,        # Decoder layers
        n_head=8,         # Number of attention heads
        head_size=32,     # Size of each attention head
        n_embd=256,       # Text embedding dimension
        img_embd_dim=512, # Visual embedding dimension
        vocab_size=vocab_size
    )

def nano_vlm_base(vocab_size=50304):
    return NanoVLM(
        n_blks=6,         # Visual encoder blocks
        n_layer=6,        # Decoder layers
        n_head=12,        # Number of attention heads
        head_size=64,     # Size of each attention head
        n_embd=512,       # Text embedding dimension
        img_embd_dim=768, # Visual embedding dimension
        vocab_size=vocab_size
    )

def nano_vlm_large(vocab_size=50304):
    return NanoVLM(
        n_blks=8,          # Visual encoder blocks
        n_layer=8,         # Decoder layers
        n_head=16,         # Number of attention heads
        head_size=64,      # Size of each attention head
        n_embd=768,        # Text embedding dimension
        img_embd_dim=1024, # Visual embedding dimension
        vocab_size=vocab_size
    )