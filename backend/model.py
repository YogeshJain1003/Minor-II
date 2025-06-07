# import matplotlib
# matplotlib.use('Agg')  # <-- Comment out or remove these lines

import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt
from cry import CryptoNet, create_ckks_context
from PIL import Image
import sys

# Function to preprocess user-supplied image
# Converts to grayscale, resizes to 28x28, normalizes
# Returns a numpy array shaped (28, 28)
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Grayscale
    image = image.resize((28, 28))
    image_np = np.array(image).astype(np.float32)
    # Normalize similar to StandardScaler used in training (mean=0, std=1)
    image_np = (image_np - np.mean(image_np)) / (np.std(image_np) + 1e-8)
    return image_np

if __name__ == "__main__":
    # Get image path from command line or set manually
    if len(sys.argv) < 2:
        print("Usage: python model.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]

    # Step 1: Load and show the original image (before preprocessing)
    original_image = Image.open(image_path).convert('L').resize((28, 28))
    plt.figure(figsize=(3,3))
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image (before preprocessing)')
    plt.axis('off')
    plt.show()

    # Step 2: Preprocess the user image
    user_img = preprocess_image(image_path)
    plt.figure(figsize=(3,3))
    plt.imshow(user_img, cmap='gray')
    plt.title('Image after preprocessing')
    plt.axis('off')
    plt.show()

    # Step 3: Create encryption context
    context = create_ckks_context()

    # Step 4: Encrypt the image (flatten to 784)
    flat_img = user_img.flatten()
    enc_img = ts.ckks_vector(context, flat_img)

    # Step 5: Load model
    model = CryptoNet()
    model.load("Cryptonet_weights_final.npz")

    # Step 6: Predict, Decrypt and Visualize
    output = model.forward(enc_img, context)
    decrypted_output = np.array(output.decrypt())
    prediction = np.argmax(decrypted_output)

    print(f"Predicted label: {prediction}")

    # Show the image after decryption (model output)
    plt.figure(figsize=(3,3))
    plt.imshow(np.array(enc_img.decrypt()).reshape(28,28), cmap='gray')
    plt.title('Image after decryption (model input)')
    plt.axis('off')
    plt.show()

    # Show the encrypted image (visualized as raw ciphertext bytes)
    enc_img_bytes = np.frombuffer(enc_img.serialize(), dtype=np.uint8)
    if enc_img_bytes.size < 784:
        enc_img_bytes = np.pad(enc_img_bytes, (0, 784 - enc_img_bytes.size), 'constant')
    elif enc_img_bytes.size > 784:
        enc_img_bytes = enc_img_bytes[:784]
    plt.figure(figsize=(3,3))
    plt.imshow(enc_img_bytes.reshape(28,28), cmap='gray')
    plt.title('Encrypted Image (raw ciphertext bytes)')
    plt.axis('off')
    plt.show()

    # Prepare all images for plotting
    original_image_np = np.array(original_image)
    preprocessed_image_np = user_img
    decrypted_image_np = np.array(enc_img.decrypt()).reshape(28,28)
    enc_img_bytes = np.frombuffer(enc_img.serialize(), dtype=np.uint8)
    if enc_img_bytes.size < 784:
        enc_img_bytes = np.pad(enc_img_bytes, (0, 784 - enc_img_bytes.size), 'constant')
    elif enc_img_bytes.size > 784:
        enc_img_bytes = enc_img_bytes[:784]
    encrypted_image_np = enc_img_bytes.reshape(28,28)

    # Plot all four images in a single figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(original_image_np, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(preprocessed_image_np, cmap='gray')
    axes[1].set_title('Preprocessed')
    axes[1].axis('off')
    axes[2].imshow(decrypted_image_np, cmap='gray')
    axes[2].set_title('Decrypted')
    axes[2].axis('off')
    axes[3].imshow(encrypted_image_np, cmap='gray')
    axes[3].set_title('Encrypted (bytes)')
    axes[3].axis('off')
    plt.tight_layout()
    plt.show()