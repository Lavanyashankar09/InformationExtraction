# # import numpy as np

# # vector = np.array([1, 0, 0])
# # matrix = np.array([[0.5, 1/12, 1/6], [0, 0, 1/9], [0, 0, 0]])

# # result = np.dot(vector, matrix)
# # print(result)


# from collections import defaultdict

# # Create a defaultdict with int as the default factory
# d = defaultdict(int)

# # Add some key-value pairs
# d['a'] = 1
# d['b'] = 2

# # Access a key that doesn't exist yet
# print(d['c'])  # Output: 0, because int() returns 0 by default

# # The dictionary now contains 'c' with a default value of 0
# print(d)  # Output: {'a': 1, 'b': 2, 'c': 0}


# # Read the contents of the file
# with open('data/clsp.trnlbls', 'r') as file:
#     data = file.read()

# # Split the contents into individual labels
# labels = data.split()

# # Count the frequency of each label
# label_counts = {}
# for label in labels:
#     if label in label_counts:
#         label_counts[label] += 1
#     else:
#         label_counts[label] = 1

# # Determine the number of unique labels
# num_unique_labels = len(label_counts)

# print("Number of unique values:", num_unique_labels)


import numpy as np
from sklearn.cluster import KMeans

# Step 1: Read the Data
with open('data/clsp.trnlbls', 'r') as file:
    data = file.read()

# Split the data into individual labels
labels = data.split()

# Step 2: Apply K-means Clustering
num_clusters = 256  # Number of clusters for vector quantization
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(np.array(labels).reshape(-1, 1))

# Get the centroids (codebook)
codebook = kmeans.cluster_centers_.flatten()

# Step 3: Quantization
# Quantize the labels using the codebook
quantized_labels = kmeans.predict(np.array(labels).reshape(-1, 1))

# Print the quantized data
print("Quantized Labels:")
print(quantized_labels)
