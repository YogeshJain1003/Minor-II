# 6. Model Accuracy vs. Number of Training Images
# Shows model accuracy as a function of the number of training images
import matplotlib.pyplot as plt

# Use your actual/estimated data
image_counts = [100, 500, 1500, 30000, 60000]
accuracies = [56.67, 76.00, 78.33, 82.0, 84.0]  # Fill in your real/estimated values

plt.figure(figsize=(8,5))
plt.plot(image_counts, accuracies, marker='o', linestyle='-', color='b')
plt.title('Model Accuracy vs. Number of Training Images')
plt.xlabel('Number of Training Images')
plt.ylabel('Accuracy (%)')
plt.xscale('log')
plt.ylim(50, 90)
# Remove grid and set x-ticks to plain digits
plt.xticks(image_counts, [str(x) for x in image_counts])
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Removed grid
for x, y in zip(image_counts, accuracies):
    plt.text(x, y+0.5, f'{y:.2f}%', ha='center')
plt.show()