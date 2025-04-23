import torch
import torch.nn as nn
import tenseal as ts
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import psutil
import time
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

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

class EncryptedDataset(torch.utils.data.Dataset):
    def __init__(self, encrypted_images, labels):
        self.encrypted_images = encrypted_images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.encrypted_images[idx], self.labels[idx]

def train_cryptonet(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the CryptoNet model"""
    model.train()
    initial_memory = get_memory_usage()
    print(f"\n📊 Initial memory usage: {initial_memory:.2f} MB")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_start_time = time.time()
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
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                current_memory = get_memory_usage()
                batch_time = time.time() - batch_start_time
                print(f"\n📈 Batch {batch_count}/{len(train_loader)}")
                print(f"⏱️ Batch time: {batch_time:.2f}s")
                print(f"📊 Current memory: {current_memory:.2f} MB")
                print(f"💾 Memory increase: {current_memory - initial_memory:.2f} MB")
        
        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        current_memory = get_memory_usage()
        
        print(f'\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'⏱️ Time: {epoch_time:.2f}s')
        print(f'📉 Loss: {epoch_loss:.4f}')
        print(f'🎯 Accuracy: {epoch_acc:.2f}%')
        print(f'💾 Memory usage: {current_memory:.2f} MB')
        print(f'📈 Memory increase: {current_memory - initial_memory:.2f} MB')

def main():
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"📊 Initial memory usage: {initial_memory:.2f} MB")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load encryption context
    print("🔑 Loading encryption context...")
    context_start_time = time.time()
    with open("tenseal_context.tenseal", "rb") as f:
        context_bytes = f.read()
    ctx = ts.context_from(context_bytes)
    print(f"✅ Context loaded in {time.time() - context_start_time:.2f}s")
    
    # Initialize model
    print("🤖 Initializing model...")
    model = CryptoNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    batch_size = 5
    
    # Load and process encrypted data in chunks
    print("\n📥 Loading and processing encrypted data...")
    all_images = []
    all_labels = []
    total_size = 0
    
    # Assuming we have chunks from 0 to N
    chunk_index = 0
    while Path(f"encrypted_mnist_vectors_{chunk_index}.pkl").exists():
        chunk_start_time = time.time()
        encrypted_images, labels = load_encrypted_data(chunk_index, ctx)
        
        # Decrypt images during loading
        decrypted_images = []
        for img in encrypted_images:
            try:
                # Create a new context with secret key for decryption
                decryption_ctx = ts.context(
                    scheme=ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=8192,
                    coeff_mod_bit_sizes=[60, 40, 40, 60]
                )
                decryption_ctx.global_scale = 2**40
                decryption_ctx.generate_galois_keys()
                
                # Decrypt the image
                encrypted_vector = ts.ckks_vector_from(decryption_ctx, img)
                decrypted_img = np.array(encrypted_vector.decrypt())
                decrypted_img = decrypted_img.reshape(1, 28, 28)
                decrypted_images.append(torch.tensor(decrypted_img, dtype=torch.float32))
            except Exception as e:
                print(f"Error decrypting image: {e}")
                continue
        
        all_images.extend(decrypted_images)
        all_labels.extend(labels)
        chunk_size = sum(img.element_size() * img.nelement() for img in decrypted_images)
        total_size += chunk_size
        chunk_time = time.time() - chunk_start_time
        
        print(f"\n📦 Chunk {chunk_index}:")
        print(f"⏱️ Load time: {chunk_time:.2f}s")
        print(f"💾 Size: {chunk_size / (1024*1024):.2f} MB")
        print(f"📊 Total size so far: {total_size / (1024*1024):.2f} MB")
        
        chunk_index += 1
    
    print(f"\n✅ Loaded {chunk_index} chunks")
    print(f"📦 Total data size: {total_size / (1024*1024):.2f} MB")
    
    # Create dataset and dataloader
    dataset = EncryptedDataset(all_images, all_labels)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Train the model
    print("\n🚀 Starting training...")
    train_cryptonet(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # Save the trained model
    print("\n💾 Saving model...")
    torch.save(model.state_dict(), "cryptonet_model.pth")
    print("✅ Model saved as cryptonet_model.pth")
    
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    print(f"\n📊 Final Statistics:")
    print(f"⏱️ Total execution time: {total_time:.2f}s")
    print(f"💾 Final memory usage: {final_memory:.2f} MB")
    print(f"📈 Total memory increase: {final_memory - initial_memory:.2f} MB")

if __name__ == "__main__":
    main() 