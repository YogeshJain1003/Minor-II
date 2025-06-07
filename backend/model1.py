import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt
from cry import CryptoNet, load_mnist_data, create_ckks_context, encrypt_data
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load and preprocess the MNIST data
X_train, X_test, y_train, y_test = load_mnist_data()
X_test, y_test = X_test[:20], y_test[:20]  # Use first 20 samples for display

# Step 2: Create encryption context
context = create_ckks_context()

# Step 3: Encrypt test data
print("Encrypting test data...")
X_test_enc, y_test_enc = encrypt_data(X_test, y_test, context)

# Step 4: Load model
model = CryptoNet()
model.load("Cryptonet_weights_final.npz")

# Step 5: Predict, Decrypt and Visualize
predictions = []
true_labels = []
probabilities = []  # Store output probabilities for each sample

for i in range(len(X_test_enc)):
    output = model.forward(X_test_enc[i], context)
    decrypted_output = np.array(output.decrypt())
    prediction = np.argmax(decrypted_output)
    true_label = y_test[i]
    predictions.append(prediction)
    true_labels.append(true_label)
    probabilities.append(decrypted_output)

    print(f"Sample {i+1}: Predicted = {prediction}, True = {true_label}")
    
    # Show the image
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Predicted: {prediction}, True: {true_label}")
    plt.axis('off')
    plt.show()

# --- Additional Graphs ---

# 1. Confusion Matrix
# Shows how often each digit is correctly/incorrectly classified
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 2. Per-Class Accuracy Bar Plot
# Shows accuracy for each digit
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(8,4))
plt.bar(range(10), per_class_acc)
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.ylim(0, 1)
plt.show()

# 3. Misclassified Samples Visualization
# Visualize images that were misclassified
misclassified_indices = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != t]
plt.figure(figsize=(10, 4))
for idx, mis_idx in enumerate(misclassified_indices[:10]):  # Show up to 10
    plt.subplot(2, 5, idx+1)
    plt.imshow(X_test[mis_idx], cmap='gray')
    plt.title(f'P:{predictions[mis_idx]}, T:{true_labels[mis_idx]}')
    plt.axis('off')
plt.suptitle('Misclassified Samples (P=Predicted, T=True)')
plt.show()

# 4. Prediction Probability Bar Plot (for first sample)
# Shows model confidence for each class on a sample
plt.figure(figsize=(8,4))
plt.bar(range(10), probabilities[0])
plt.xlabel('Digit')
plt.ylabel('Model Output (Confidence)')
plt.title('Prediction Probabilities for First Sample')
plt.show()

# 5. Prediction Distribution Histogram
# Shows how often each digit is predicted
plt.figure(figsize=(8,4))
plt.hist(predictions, bins=np.arange(11)-0.5, rwidth=0.8)
plt.xlabel('Predicted Digit')
plt.ylabel('Count')
plt.title('Prediction Distribution')
plt.xticks(range(10))
plt.show()

