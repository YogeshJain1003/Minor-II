import torch
import torch.nn as nn
import tenseal as ts
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

class CryptoNet(nn.Module):
    def __init__(self):
        super(CryptoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 160)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_encrypted_data(chunk_index, ctx):
    """Load encrypted data from a specific chunk"""
    with open(f"encrypted_mnist_vectors_{chunk_index}.pkl", "rb") as f:
        encrypted_images = pickle.load(f)
    labels = torch.load(f"mnist_labels_{chunk_index}.pt")
    return encrypted_images, labels

def train_cryptonet(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the CryptoNet model"""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load encryption context
    print("Loading encryption context...")
    with open("tenseal_context.tenseal", "rb") as f:
        context_bytes = f.read()
    ctx = ts.context_from(context_bytes)
    
    # Initialize model
    model = CryptoNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    batch_size = 64
    
    # Load and process encrypted data in chunks
    print("Loading and processing encrypted data...")
    all_images = []
    all_labels = []
    
    # Assuming we have chunks from 0 to N
    chunk_index = 0
    while Path(f"encrypted_mnist_vectors_{chunk_index}.pkl").exists():
        encrypted_images, labels = load_encrypted_data(chunk_index, ctx)
        all_images.extend(encrypted_images)
        all_labels.extend(labels)
        chunk_index += 1
    
    # Convert to tensors
    labels_tensor = torch.tensor(all_labels)
    
    # Create dataset and dataloader
    class EncryptedDataset(torch.utils.data.Dataset):
        def __init__(self, encrypted_images, labels, ctx):
            self.encrypted_images = encrypted_images
            self.labels = labels
            self.ctx = ctx
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            # Decrypt the image (in practice, you'd keep it encrypted)
            # This is just for demonstration
            encrypted_vector = ts.ckks_vector_from(self.ctx, self.encrypted_images[idx])
            image = np.array(encrypted_vector.decrypt())
            image = image.reshape(1, 28, 28)  # Reshape to image format
            return torch.tensor(image, dtype=torch.float32), self.labels[idx]
    
    # Create dataset and dataloader
    dataset = EncryptedDataset(all_images, labels_tensor, ctx)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    # Train the model
    print("Starting training...")
    train_cryptonet(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), "cryptonet_model.pth")
    print("Model saved as cryptonet_model.pth")

if __name__ == "__main__":
    main() 