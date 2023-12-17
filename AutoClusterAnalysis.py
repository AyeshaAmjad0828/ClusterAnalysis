import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, adjusted_mutual_info_score, \
    adjusted_rand_score, fowlkes_mallows_score
from scipy.spatial.distance import cdist

# List of dataset file paths
dataset_files = [
    'Sources/WithLabels/HeartDisease_processed_selected_features.csv',
    'Sources/WithLabels/BankMarketing_processed_selected_features.csv',
    'Sources/WithLabels/HeartFailure_processed_selected_features.csv',
    'Sources/WithoutLabels/CarEvaluation_processed_selected_features.csv',
    'Sources/WithoutLabels/Mushroom_processed_selected_features.csv',
    'Sources/WithoutLabels/OnlinePurchasingIntentions_processed_selected_features.csv'
]

# Function to perform K-Means clustering and return optimal cluster count and centroids
def kmeans_clustering(data):
    silhouette_scores = []
    cluster_counts = []
    centroids = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_score = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_score)
        cluster_counts.append(k)
        centroids.append(kmeans.cluster_centers_)

    # Find the index of maximum silhouette score
    optimal_cluster_index = silhouette_scores.index(max(silhouette_scores))
    optimal_cluster_count = cluster_counts[optimal_cluster_index]
    optimal_centroids = centroids[optimal_cluster_index]

    return optimal_cluster_count, optimal_centroids


# Function to perform Agglomerative clustering and return optimal cluster count
def agglomerative_clustering(data):
    silhouette_scores = []
    cluster_counts = []

    for k in range(2, 11):
        agglomerative = AgglomerativeClustering(n_clusters=k)
        agglomerative.fit(data)
        labels = agglomerative.labels_
        silhouette_score = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_score)
        cluster_counts.append(k)

    # Find the index of maximum silhouette score
    optimal_cluster_index = silhouette_scores.index(max(silhouette_scores))
    optimal_cluster_count = cluster_counts[optimal_cluster_index]

    return optimal_cluster_count


# Function to perform DBSCAN clustering and return optimal cluster count
def dbscan_clustering(data):
    silhouettes = []
    cluster_counts = []

    for eps in range(1, 10):
        eps = eps / 10
        for min_samples in range(2, 6):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(data)
            labels = dbscan.labels_
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            if len(unique_labels) > 1:
                silhouette = silhouette_score(data, labels)
                silhouettes.append(silhouette)
                cluster_counts.append(len(unique_labels))

    # Find the index of maximum silhouette score
    optimal_cluster_index = silhouettes.index(max(silhouettes))
    optimal_cluster_count = cluster_counts[optimal_cluster_index]

    return optimal_cluster_count


# Function to compute performance index for optimal clusters
def compute_performance_index(true_labels, predicted_labels):
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    fm = fowlkes_mallows_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    return ami, fm, ari

# Function to display cluster centroids
def display_cluster_centroids(centroids):
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i+1} centroid: {centroid}")

# Function to compute metrics for optimal cluster selection
def compute_metrics(data, true_labels):
    kmeans_cluster_count, kmeans_centroids = kmeans_clustering(data)
    agglomerative_cluster_count = agglomerative_clustering(data)
    dbscan_cluster_count = dbscan_clustering(data)

    print("K-Means:")
    print(f"Optimal cluster count: {kmeans_cluster_count}")
    display_cluster_centroids(kmeans_centroids)

    print("\nAgglomerative:")
    print(f"Optimal cluster count: {agglomerative_cluster_count}")

    print("\nDBSCAN:")
    print(f"Optimal cluster count: {dbscan_cluster_count}")

    # Compute performance index for K-Means clustering
    kmeans = KMeans(n_clusters=kmeans_cluster_count, random_state=0)
    kmeans.fit(data)
    kmeans_labels = kmeans.labels_
    ami, fm, ari = compute_performance_index(true_labels, kmeans_labels)
    print("\nPerformance Index - K-Means")
    print(f"Adjusted Mutual Info: {ami}")
    print(f"Fowlkes-Mallows: {fm}")
    print(f"Adjusted Rand Index: {ari}")

    # Perform Agglomerative clustering with optimal cluster count
    agglomerative = AgglomerativeClustering(n_clusters=agglomerative_cluster_count)
    agglomerative.fit(data)
    agglomerative_labels = agglomerative.labels_
    ami, fm, ari = compute_performance_index(true_labels, agglomerative_labels)
    print("\nPerformance Index - Agglomerative")
    print(f"Adjusted Mutual Info: {ami}")
    print(f"Fowlkes-Mallows: {fm}")
    print(f"Adjusted Rand Index: {ari}")

    # Perform DBSCAN clustering with optimal cluster count
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan.fit(data)
    dbscan_labels = dbscan.labels_
    unique_dbscan_labels = set(dbscan_labels)
    if -1 in unique_dbscan_labels:
        unique_dbscan_labels.remove(-1)
    if len(unique_dbscan_labels) > 1:
        ami, fm, ari = compute_performance_index(true_labels, dbscan_labels)
        print("\nPerformance Index - DBSCAN")
        print(f"Adjusted Mutual Info: {ami}")
        print(f"Fowlkes-Mallows: {fm}")
        print(f"Adjusted Rand Index: {ari}")


# Iterate through the datasets
for i in range(1, 11):
    dataset_path = f"dataset_{i}.csv"
    true_labels_path = f"true_labels_{i}.csv"

    # Load the dataset and true labels
    data = pd.read_csv(dataset_path)
    true_labels = pd.read_csv(true_labels_path)

    # Perform clustering analysis and compute metrics
    compute_metrics(data, true_labels)
