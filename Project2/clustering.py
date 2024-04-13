import numpy as np
from sklearn.cluster import KMeans

# Step 1: Read the Data
with open('data/clsp.trnlbls', 'r') as file:
    data = file.readlines()[1:]  # Ignore the first line

# Combine lines into a single string and split into individual labels
labels = ''.join(data).split()

# Convert string labels to numerical values
unique_labels = list(set(labels))  # Get unique labels
label_to_index = {label: i for i, label in enumerate(unique_labels)}  # Map each label to an index
numeric_labels = [label_to_index[label] for label in labels]  # Convert labels to numerical values

# Step 2: Apply K-means Clustering
num_clusters = 256  # Number of clusters for vector quantization
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(np.array(numeric_labels).reshape(-1, 1))

# Get the centroids (codebook)
codebook = kmeans.cluster_centers_.flatten()

# Step 3: Quantization
# Quantize the labels using the codebook
quantized_labels = kmeans.predict(np.array(numeric_labels).reshape(-1, 1))

# Convert quantized numerical labels back to string representations
quantized_strings = [unique_labels[idx] for idx in quantized_labels]

# Print the quantized data
print("Quantized Labels:")
print(quantized_strings)
