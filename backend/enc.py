import torch
import torchvision
import torchvision.transforms as transforms
import tenseal as ts
from tqdm import tqdm
import os
import shutil
from pathlib import Path
import pickle

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    # Remove any existing encrypted files
    for file in os.listdir('.'):
        if file.startswith('encrypted_mnist_vectors_') or file.startswith('mnist_labels_') or file == 'tenseal_context.tenseal':
            os.remove(file)
    if os.path.exists('data'):
        shutil.rmtree('data')

def create_tenseal_context():
    """Create and return a TenSEAL CKKS context"""
    try:
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    except Exception as e:
        print(f"Error creating TenSEAL context: {e}")
        return None

def encrypt_mnist(chunk_size=10):
    """
    Encrypt MNIST dataset in chunks
    Args:
        chunk_size: Number of images per chunk (default: 10)
    """
    # Clean up any existing files
    cleanup_temp_files()

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Load full MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        print("📥 Loading MNIST dataset...")
        dataset = torchvision.datasets.MNIST(root=str(data_dir), train=True, transform=transform, download=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)

        # Create TenSEAL context
        print("🔑 Creating encryption context...")
        ctx = create_tenseal_context()
        if ctx is None:
            raise Exception("Failed to create TenSEAL context")

        # Encrypt the MNIST images in chunks
        current_chunk = []
        current_labels = []
        chunk_index = 0
        total_images = 0

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="🔐 Encrypting MNIST", total=len(data_loader)):
                try:
                    images = images.view(images.size(0), -1)  # Flatten each image to 784
                    for i in range(images.size(0)):
                        enc_image = ts.ckks_vector(ctx, images[i].tolist())
                        current_chunk.append(enc_image.serialize())
                        current_labels.append(targets[i].item())
                        total_images += 1

                        # Save chunk when it reaches the desired size
                        if len(current_chunk) >= chunk_size:
                            # Save encrypted chunk
                            chunk_filename = f"encrypted_mnist_vectors_{chunk_index}.pkl"
                            with open(chunk_filename, "wb") as f:
                                pickle.dump(current_chunk, f)
                            
                            # Save corresponding labels
                            labels_filename = f"mnist_labels_{chunk_index}.pt"
                            torch.save(current_labels, labels_filename)
                            
                            print(f"✅ Saved chunk {chunk_index} with {len(current_chunk)} images")
                            
                            # Reset for next chunk
                            current_chunk = []
                            current_labels = []
                            chunk_index += 1

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        # Save any remaining images in the last chunk
        if current_chunk:
            chunk_filename = f"encrypted_mnist_vectors_{chunk_index}.pkl"
            with open(chunk_filename, "wb") as f:
                pickle.dump(current_chunk, f)
            
            labels_filename = f"mnist_labels_{chunk_index}.pt"
            torch.save(current_labels, labels_filename)
            
            print(f"✅ Saved final chunk {chunk_index} with {len(current_chunk)} images")
        
        # Save the context
        with open("tenseal_context.tenseal", "wb") as f:
            f.write(ctx.serialize())

        print(f"✅ All {total_images} MNIST images encrypted and saved in {chunk_index + 1} chunks!")
        print("📁 Files created:")
        print(f"- encrypted_mnist_vectors_*.pkl (encrypted images in {chunk_index + 1} chunks)")
        print(f"- mnist_labels_*.pt (corresponding labels in {chunk_index + 1} chunks)")
        print("- tenseal_context.tenseal (encryption context)")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        cleanup_temp_files()
        raise

if __name__ == "__main__":
    # You can adjust the chunk_size here (default is 10 images per chunk)
    encrypt_mnist(chunk_size=10)
