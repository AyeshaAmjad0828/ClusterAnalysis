import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
import skfuzzy as fuzz
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, adjusted_mutual_info_score, \
    adjusted_rand_score, fowlkes_mallows_score
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import os

# List of dataset file paths
dataset_files = [
    # 'Sources/WithLabels/HeartDisease_processed_selected_features.csv',
    # 'Sources/WithLabels/BankMarketing_processed_selected_features.csv',
    'Sources/WithLabels/HeartFailure_processed_selected_features.csv'
    # 'Sources/WithoutLabels/CarEvaluation_processed_selected_features.csv',
    # 'Sources/WithoutLabels/Mushroom_processed_selected_features.csv',
    # 'Sources/WithoutLabels/OnlinePurchasingIntentions_processed_selected_features.csv'
]

# file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in dataset_files]

# print(file_names)

def calculate_internal_metrics(X, labels):
    silhouette = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    # Compute Dunn Index
    dists = euclidean_distances(X)
    min_cluster_dists = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                min_cluster_dists.append(dists[i][j])
    min_inter_cluster_distance = np.min(min_cluster_dists)
    max_intra_cluster_diameter = max([np.max(dists[np.where(labels == label)][:, np.where(labels == label)]) for label in np.unique(labels)])
    dunn_index = min_inter_cluster_distance / max_intra_cluster_diameter

    calin_harab = calinski_harabasz_score(X, labels)

    return silhouette, dbi, dunn_index, calin_harab


def calculate_external_metrics(labels_true, labels_pred):
    adj_mutual_info = adjusted_mutual_info_score(labels_true, labels_pred)
    adj_rand_index = adjusted_rand_score(labels_true, labels_pred)
    fowlkes_mallows = fowlkes_mallows_score(labels_true, labels_pred)

    return adj_mutual_info, adj_rand_index, fowlkes_mallows


def GMM_Analysis(X, max_clusters):

    optimal_clusters_gmm = 0
    max_silhouette = -1
    gmm_labels = None 

    for n_clusters in range(2, max_clusters + 1):
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters)
        gmm_labels = gmm.fit_predict(X)
        gmm_metrics = calculate_internal_metrics(X, gmm_labels)
        silhouette_avg_gmm = silhouette_score(X, gmm_labels)
    
        if silhouette_avg_gmm > max_silhouette:
            max_silhouette = silhouette_avg_gmm
            optimal_clusters_gmm = n_clusters


    return gmm_metrics, silhouette_avg_gmm, optimal_clusters_gmm, gmm_labels


def FCM_Analysis(X, max_clusters):

    optimal_clusters_fuzzy = 0
    max_silhouette = -1
    fuzzy_labels = None 
    max_clusters = 5

    for n_clusters in range(2, max_clusters + 1):
    
        fuzzy_result = fuzz.cluster.cmeans(
            data.T, n_clusters, 2, error=0.005, maxiter=1000
        )

        # Getting fuzzy centroids and membership matrix
        fuzzy_centroids = fuzzy_result[0]
        fuzzy_membership = fuzzy_result[1]

        # Getting cluster labels
        fuzzy_labels = np.argmax(fuzzy_membership, axis=0)

        fuzzy_metrics = calculate_internal_metrics(data, fuzzy_labels)
        silhouette_avg_fuzzy= silhouette_score(data, fuzzy_labels)
    
        if silhouette_avg_fuzzy > max_silhouette:
            max_silhouette = silhouette_avg_fuzzy
            optimal_clusters_fuzzy = n_clusters


    return fuzzy_metrics, silhouette_avg_fuzzy, optimal_clusters_fuzzy, fuzzy_labels

def Spectral_Analysis(X, max_clusters):

    optimal_clusters_spectral = 0
    max_silhouette = -1
    spectral_labels = None 

    for n_clusters in range(2, max_clusters + 1):
    
        # Spectral Clustering
        spectral = SpectralClustering(n_clusters=n_clusters)
        spectral_labels = spectral.fit_predict(X)
        spectral_metrics = calculate_internal_metrics(X, spectral_labels)
        silhouette_avg_spectral = silhouette_score(X, spectral_labels)

        if silhouette_avg_spectral > max_silhouette:
            max_silhouette = silhouette_avg_spectral
            optimal_clusters_spectral = n_clusters


    return  spectral_metrics, silhouette_avg_spectral, optimal_clusters_spectral, spectral_labels




for dataset in dataset_files:
    # Load the current dataset
    file_name = os.path.splitext(os.path.basename(dataset))[0]
    data = pd.read_csv(dataset)
    gmm_metrics, silhouette_avg_gmm, optimal_clusters_gmm, gmm_labels = GMM_Analysis(data, max_clusters=5)
    fuzzy_metrics, silhouette_avg_fuzzy, optimal_clusters_fuzzy, fuzzy_labels = FCM_Analysis(data, max_clusters=5)
    spectral_metrics, silhouette_avg_spectral, optimal_clusters_spectral, spectral_labels = Spectral_Analysis(data, max_clusters=5)

    print(f"GMM Metrics (silhouette, dbi, dunn_index, calin_harab) for {file_name} : {gmm_metrics}", f",Optimal Clusters: {optimal_clusters_gmm}", f",Average Silhouette Score: {silhouette_avg_gmm}")
    print(f"FCM Metrics (silhouette, dbi, dunn_index, calin_harab) for {file_name} : {fuzzy_metrics}", f",Optimal Clusters: {optimal_clusters_fuzzy}", f",Average Silhouette Score: {silhouette_avg_fuzzy}")
    print(f"Spectral Metrics (silhouette, dbi, dunn_index, calin_harab) for {file_name} : {spectral_metrics}", f",Optimal Clusters: {optimal_clusters_spectral}", f",Average Silhouette Score: {silhouette_avg_spectral}")

    #print(f"Metrics and Optimal Clusters for {dataset}: {clustering_result}")

    if 'target' in data.columns:
        X = data.drop('target', axis=1)  # I have changed the header of dependent variable in each dataset to "target"
        y = data['target']
        gmm_adj_mutual_info, gmm_adj_rand_index, gmm_fowlkes_mallows = calculate_external_metrics(y, gmm_labels)
        fcm_adj_mutual_info, fcm_adj_rand_index, fcm_fowlkes_mallows = calculate_external_metrics(y, fuzzy_labels)
        spectral_adj_mutual_info, spectral_adj_rand_index, spectral_fowlkes_mallows = calculate_external_metrics(y, spectral_labels)
        
        print(f"GMM External Metrics (AMI, ARI, FM) for {file_name} : {gmm_adj_mutual_info}, {gmm_adj_rand_index}, {gmm_fowlkes_mallows}")
        print(f"FCM External Metrics (AMI, ARI, FM) for {file_name} : {fcm_adj_mutual_info}, {fcm_adj_rand_index}, {fcm_fowlkes_mallows}")
        print(f"Spectral External Metrics (AMI, ARI, FM) for {file_name} : {spectral_adj_mutual_info}, {spectral_adj_rand_index}, {spectral_fowlkes_mallows}")






