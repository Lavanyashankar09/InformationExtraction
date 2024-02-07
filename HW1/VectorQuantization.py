# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to perform vector quantization
def vector_quantize(dp_numpy, n_clusters):

    # Initialize centroids randomly
    centeroids = dp_numpy[np.random.choice(len(dp_numpy), n_clusters, replace=False)]
    #initial_centroid_indices = [0, 50, 99]  # Adjust these indices based on your preferences
    #centeroids = dp_numpy[initial_centroid_indices]
    print("Initial Centers:", centeroids)
    
    # Initialize empty list for cluster labels and set iteration count to 0
    old_label = []
    iteration_count = 0
    
    # Iterate until convergence
    while True:
        # Initialize empty list for new cluster labels
        new_label = []
        
        # Assign each vector to the nearest centroid

        for each_dp in dp_numpy:
            distances = []
            
            for each_c in centeroids:
                distance = np.linalg.norm(each_dp - each_c)
                distances.append(distance)
            
            closest_center_index = np.argmin(distances)
            new_label.append(closest_center_index)

        
        # Check for convergence by comparing with previous labels
        if np.array_equal(old_label, new_label):
            break
        
        # Update labels and increment iteration count
        old_label = new_label
        iteration_count += 1

        # Plot the current iteration
        plot_iteration(dp_numpy, old_label, centeroids, iteration_count)
        
        # Update centroids based on mean of points in each cluster
        for cluster_index in range(n_clusters):
            current_cluster_points = []

            for data_point_index in range(len(dp_numpy)):
                if old_label[data_point_index] == cluster_index:
                    current_cluster_points.append(dp_numpy[data_point_index])

            if len(current_cluster_points) == 0:
                continue

            current_cluster_points = np.array(current_cluster_points)
            centeroids[cluster_index, :] = np.mean(current_cluster_points, axis=0)
    
    # Print the number of iterations and return the final labels and centroids
    print(f'Number of iterations: {iteration_count}')
    return old_label, centeroids

# Function to plot the current iteration
def plot_iteration(vectors, old_label, centers, iteration_count):
    print("Centers:", centers)
    plt.figure()
    plt.scatter(vectors[:, 0], vectors[:, 1], c=old_label, cmap='viridis', marker='o', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title(f'Iteration {iteration_count}')
    plt.legend()
    plt.show()

# Main function
def main():
    # Read data from file
    f = open("/Users/lavanya/Library/CloudStorage/OneDrive-JohnsHopkins/Courses/Spring2024/InformationExtraction/HW1/hw1-data (1).txt", "r")
    cluster = 3
    dp_list = []
    
    # Parse data and convert to numpy array
    for line in f:
        line = line.strip().split()
        dp_list.append((float(line[0]), float(line[1])))
    
    dp_numpy = np.array(dp_list)
    
    # Perform vector quantization
    vector_quantize(dp_numpy, cluster)

# Execute main function if the script is run as the main program
if __name__ == "__main__":
    main()
