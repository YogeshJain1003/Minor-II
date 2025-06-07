import tenseal as ts
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas
from tqdm import tqdm
import os

# Load and preprocess MNIST dataset
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = StandardScaler().fit_transform(X)  
    X = X.reshape(-1, 28, 28)  
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# CKKS context setup for encryption
def create_ckks_context():
    poly_mod_degree = 16384  
    coeff_mod_bits = [60, 40, 40, 40, 40, 60]  
    context = ts.Context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bits
    )
    context.global_scale = 2**30  
    context.generate_galois_keys()
    return context

# Encrypt images and labels
def encrypt_data(X, y, context):
    encrypted_X = []
    encrypted_y = []
    for i in tqdm(range(len(X)), desc="Encrypting data"):
        flat_img = X[i].flatten()
        enc_img = ts.ckks_vector(context, flat_img)
        encrypted_X.append(enc_img)
        one_hot = np.zeros(10)
        one_hot[y[i]] = 1.0
        enc_label = ts.ckks_vector(context, one_hot)
        encrypted_y.append(enc_label)
    return encrypted_X, encrypted_y

#  CryptoNet model with polynomial activation and checkpointing
class CryptoNet:
    def __init__(self, input_dim=784, output_dim=10):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
    
    def forward(self, x, context):
        out = x.matmul(self.W) + self.b
        return out * out 
    
    def compute_loss(self, output, target):
        diff = output - target
        return (diff * diff).sum() * (1.0 / output.size())
    
    def backward(self, x, target, output, context, lr=0.01):
        diff = output - target
        grad_W = np.outer(x.decrypt(), diff.decrypt())  
        grad_b = np.array(diff.decrypt())
        self.W -= lr * grad_W
        self.b -= lr * grad_b

    def save(self, filename):
        np.savez(filename, W=self.W, b=self.b)

    def load(self, filename):
        data = np.load(filename)
        self.W = data['W']
        self.b = data['b']

# Training loop with checkpointing
def train(model, X_enc, y_enc, context, epochs=1, lr=0.01, start_epoch=0, checkpoint_prefix='cryptonet_weights_epoch'):
    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        for i in tqdm(range(len(X_enc)), desc=f"Training Epoch {epoch+1}"):
            output = model.forward(X_enc[i], context)
            loss = model.compute_loss(output, y_enc[i])
            total_loss += loss.decrypt()[0]
            model.backward(X_enc[i], y_enc[i], output, context, lr)
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(X_enc)}")
        # Save checkpoint after each epoch
        model.save(f'{checkpoint_prefix}{epoch+1}.npz')

# Evaluation
def evaluate(model, X_enc, y_enc, context):
    correct = 0
    for i in tqdm(range(len(X_enc)), desc="Testing"):
        output = model.forward(X_enc[i], context)
        pred = np.argmax(np.array(output.decrypt()))
        true = np.argmax(np.array(y_enc[i].decrypt()))
        if pred == true:
            correct += 1
    accuracy = correct / len(X_enc)
    return accuracy

# Main execution 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start/resume training from')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')
    args = parser.parse_args()

    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()

    # Combine train and test for per-label selection
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    X_train_sel, y_train_sel = [], []
    X_test_sel, y_test_sel = [], []

    for label in range(10):
        idx = np.where(y_all == label)[0]
        #np.random.shuffle(idx)
        train_idx = idx[:150]
        test_idx = idx[150:180]
        X_train_sel.append(X_all[train_idx])
        y_train_sel.append(y_all[train_idx])
        X_test_sel.append(X_all[test_idx])
        y_test_sel.append(y_all[test_idx])

    X_train = np.concatenate(X_train_sel, axis=0)
    y_train = np.concatenate(y_train_sel, axis=0)
    X_test = np.concatenate(X_test_sel, axis=0)
    y_test = np.concatenate(y_test_sel, axis=0)

    print("Selected", len(X_train), "training images and", len(X_test), "testing images.")

    # Setup CKKS context
    context = create_ckks_context()

    # Encrypt data
    print("Encrypting training data...")
    X_train_enc, y_train_enc = encrypt_data(X_train, y_train, context)
    print("Encrypting test data...")
    X_test_enc, y_test_enc = encrypt_data(X_test, y_test, context)

    # Initialize model
    model = CryptoNet()
    if args.resume_checkpoint is not None and os.path.exists(args.resume_checkpoint):
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        model.load(args.resume_checkpoint)

    # Train model
    print("Training model...")
    train(model, X_train_enc, y_train_enc, context, epochs=20-args.start_epoch, lr=0.01, start_epoch=args.start_epoch)

    # Evaluate model
    print("Evaluating model...")
    accuracy = evaluate(model, X_test_enc, y_test_enc, context)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save final model
    model.save('cryptonet_weights_final.npz')
