# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV



"""
    Perform automated cluster analysis on a given dataset.

    Parameters:
    - data (DataFrame): The dataset containing features and the target column.
    - target_col (str): The name of the target column.
    - num_clusters_range (list): A list of the number of clusters to consider.
    - scaling_method (str): 'standardize' or 'normalize'.
    - feature_selection (bool): Whether to perform feature selection.

    Returns:
    - best_cluster_count (int): The optimal number of clusters.
    - cluster_labels (array): Cluster labels for each data point.
    - cluster_centers (array): Centroids of the clusters.
    - selected_features (list): List of selected feature names.
    """

#identify and drop contant value columns
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(constant_columns, axis=1)


#identify and drop sequential columns
sequential_columns = []
for col in df.columns:
    if df[col].dtype in [np.int64, np.int32, np.float64]:
        differences = np.diff(df[col])
        if np.all(differences == differences[0]):
            sequential_columns.append(col)

df = df.drop(sequential_columns, axis=1)



#deal with missing values
df = df.replace(r'^\s*$', pd.NA, regex=True)

#identify numeric and categorical variables
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=np.object).columns

# Mean imputation for numeric fields and Mode imputation for categorical fields
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

def perform_cluster_analysis(data, target_col, num_clusters_range, scaling_method='standardize', feature_selection=True):
 

    # Step 1: Data Preprocessing
    X = data.drop(columns=[target_col])
    y = data[target_col]

    

    # Step 2: Feature Scaling
    if scaling_method == 'standardize':
        scaler = StandardScaler()
    elif scaling_method == 'normalize':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Use 'standardize' or 'normalize'.")
    
    X_scaled = scaler.fit_transform(X)

    # Step 3: Feature Selection (if enabled)
    if feature_selection:
        selector = SelectKBest(f_classif, k='all')
        X_scaled = selector.fit_transform(X_scaled, y)
        selected_features = [X.columns[i] for i in np.argsort(selector.scores_)[::-1]]
    else:
        selected_features = X.columns.tolist()

    # Step 4: Hyperparameter Optimization
    best_cluster_count = None
    best_silhouette_score = -1

    for n_clusters in num_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_cluster_count = n_clusters
            cluster_centers = kmeans.cluster_centers_
    
    # Step 5: Intrinsic Evaluation (Davies-Bouldin Score)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    # Step 6: Display Cluster Centroids
    for i, centroid in enumerate(cluster_centers):
        print(f"Cluster {i} Centroid (scaled): {centroid}")

    # Step 7: Visualize the Clusters (for 2D data)
    if X_scaled.shape[1] == 2:
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=100, label='Centroids')
        plt.title(f'Clustering Results (Silhouette Score: {best_silhouette_score:.2f}, DB Score: {db_score:.2f})')
        plt.legend()
        plt.show()

    return best_cluster_count, cluster_labels, cluster_centers, selected_features

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('your_dataset.csv')

    # Define your target column and range of cluster counts
    target_column = 'target'
    cluster_count_range = range(2, 11)  # Example range from 2 to 10 clusters

    # Perform cluster analysis
    best_clusters, cluster_labels, centroids, selected_features = perform_cluster_analysis(data, target_column, cluster_count_range)

    print(f"Optimal Number of Clusters: {best_clusters}")
    print(f"Selected Features: {', '.join(selected_features)}")
