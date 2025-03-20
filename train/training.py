import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Training function
def train_nano_vlm(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    """
    Training function for NanoVLM
    
    Args:
        model: NanoVLM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            imgs, text_tokens, targets = batch
            imgs = imgs.to(device)
            text_tokens = text_tokens.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(imgs, text_tokens)
            
            # Calculate loss (ignore the visual token in the output)
            loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, text_tokens, targets = batch
                imgs = imgs.to(device)
                text_tokens = text_tokens.to(device)
                targets = targets.to(device)
                
                # Forward pass
                logits = model(imgs, text_tokens)
                
                # Calculate loss
                loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
        
        # Record losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


# Function to plot training and validation losses
def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses of NanoVLM')
    plt.legend()
    plt.grid(True)
    plt.show()