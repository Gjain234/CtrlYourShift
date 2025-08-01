import numpy as np
import pandas as pd

# Sigmoid function to map values to [0, 1]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to generate the heterogeneous dataset
def generate_heterogeneous_data(total_size, min_group_size, n_groups, feature_dim=5, noise_scale=0.05,global_weight_scale=0.5,seed=42):
    np.random.seed(seed)
    # Step 1: Generate imbalanced group sizes using Pareto distribution
    group_sizes = np.random.pareto(a=2, size=n_groups) * 50  # Highly imbalanced distribution
    group_sizes = group_sizes / group_sizes.sum()  # Normalize so that they sum up to 1
    group_sizes = np.round(group_sizes * total_size).astype(int)  # Scale to total size and round to integers

    # Step 2: Ensure the smallest group size is at least min_group_size
    while min(group_sizes) < min_group_size:
        group_sizes = np.random.pareto(a=2, size=n_groups) * 50  # Recompute group sizes
        group_sizes = group_sizes / group_sizes.sum()  # Normalize again
        group_sizes = np.round(group_sizes * total_size).astype(int)  # Scale to total size and round

    # Step 3: Ensure the total size matches
    # Fix rounding errors by adjusting the last group size
    group_sizes_sum = group_sizes.sum()
    if group_sizes_sum != total_size:
        group_sizes[-1] += total_size - group_sizes_sum

    # Step 4: Generate random continuous features
    features = np.random.randn(total_size, feature_dim)  # Continuous features

    # Step 5: Assign group labels based on the group sizes
    group_labels = []
    for i, size in enumerate(group_sizes):
        group_labels.extend([f'group_{i}'] * size)  # Assign each individual to the corresponding group
    group_labels = np.array(group_labels)

    n_cluster_groups = int(n_groups / 2)  # 2/3 of the groups will have clusters
    group_weights = np.zeros((n_groups, feature_dim))

    # Randomly select 2/3 of the groups for clustering
    cluster_groups = np.random.choice(n_groups, size=n_cluster_groups, replace=False)

    # Step 7: Create clusters by combining multiple groups
    # Now use a while loop to cluster groups with random sizes between 2 and 7
    all_group_indices = list(cluster_groups)
    np.random.shuffle(all_group_indices)  # Shuffle the indices of groups to create clusters

    clusters = []
    idx = 0
    while idx < len(all_group_indices):
        cluster_size = np.random.randint(2, 8)  # Random cluster size between 2 and 7
        cluster = all_group_indices[idx:idx + cluster_size]
        clusters.append(cluster)
        idx += cluster_size

    # Step 8: Assign a shared weight for each cluster and small random shifts
    for cluster in clusters:
        # Generate a random base weight for the cluster
        base_weight = np.random.randn(feature_dim)

        # For each group in the cluster, assign the shared base weight with small shifts
        for group_idx in cluster:
            # Generate small random shifts for each individual within this cluster
            group_mask = group_labels == f'group_{group_idx}'
            shift = np.random.normal(scale=0.1, size=feature_dim)  # Small random shift
            group_weights[group_idx] = base_weight + shift  # Assign the shifted weight for this group

    # Step 9: For the remaining groups, generate weights based on a distribution
    remaining_groups = list(set(range(n_groups)) - set(cluster_groups))

    # Generate a random mean and standard deviation for each feature for the remaining groups
    feature_means = np.random.randn(feature_dim)  # Means for each feature
    feature_stds = np.random.rand(feature_dim) * 0.5  # Standard deviations for each feature, scaled to avoid too large values

    for group_idx in remaining_groups:
        # Sample new weights for each feature based on the means and stds
        group_weights[group_idx] = np.random.normal(loc=feature_means, scale=feature_stds)

    # Step 7: Compute the local probabilities (for each individual within each group)
    local_probs = np.zeros(total_size)
    for i in range(n_groups):
        group_mask = group_labels == f'group_{i}'
        group_feature_sum = np.dot(features[group_mask], group_weights[i])  # Weighted sum of features for each individual
        local_probs[group_mask] = group_feature_sum + np.random.normal(0, noise_scale, np.sum(group_mask))  # Add noise

    # Step 8: Generate global weights (for individual-level probabilities)
    global_weights = np.random.randn(feature_dim)

    # Step 9: Compute the global probabilities (for each individual)
    global_probs = np.dot(features, global_weights)  # Weighted sum of features for individuals
    global_probs = global_probs + np.random.normal(0, noise_scale, total_size)  # Add noise

    # Step 10: Combine the global and local probabilities using a weighted sum
    outcome_probs = global_weight_scale * global_probs + (1-global_weight_scale) * local_probs  # You can adjust the weights here
    # Apply the sigmoid function to ensure probabilities are within [0, 1]
    outcome_probs = sigmoid(outcome_probs)
    # Step 11: Generate binary outcomes based on the computed probabilities
    outcomes = np.random.rand(total_size) < outcome_probs

    # Step 12: Combine everything into a DataFrame
    data = pd.DataFrame(features, columns=[f'feature_{i+1}' for i in range(feature_dim)])
    data['group'] = group_labels
    data['outcome'] = outcomes.astype(int)
    # Step 12: Create a matrix for predicted probabilities across all groups
    group_prob_matrix = np.zeros((total_size, n_groups))  # Initialize matrix for probabilities
    for i in range(n_groups):
        # For each group, calculate the predicted probabilities for each individual
        group_feature_sum = np.dot(features, group_weights[i])  # Weighted sum of features for the group
        group_local_probs = group_feature_sum + np.random.normal(0, noise_scale, total_size)  # Addnoise
        group_global_probs = np.dot(features, global_weights)  # Global prediction for group
        group_global_probs = group_global_probs + np.random.normal(0, noise_scale, total_size)  # Add noise
        group_combined_probs = global_weight_scale * group_global_probs + (1-global_weight_scale) * group_local_probs  # Combine probabilities
        group_combined_probs = sigmoid(group_combined_probs)  # Ensure between 0 and 1
        group_prob_matrix[:, i] = group_combined_probs  # Store in matrix
    group_prob_df = pd.DataFrame(group_prob_matrix, columns=[f'group_{i}' for i in range(n_groups)])
    return data,group_prob_df

# Example usage
total_size = 40000  # Total number of samples
min_group_size = 120  # Minimum group size
n_groups = 50  # Number of groups
feature_dim = 20

# Generate the dataset
dataset,group_prob_df = generate_heterogeneous_data(total_size, min_group_size, n_groups, feature_dim,global_weight_scale=0.3)
dataset.to_csv('data/synthetic_dataset.csv', index=False)
