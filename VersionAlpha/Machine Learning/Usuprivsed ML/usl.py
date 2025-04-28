import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
import umap.umap_ as umap

def load_data(filepath):
    """Load and preprocess the fault classification data."""
    df = pd.read_csv(filepath)
    
    # Combine binary outputs (G, C, B, A) into a single fault type label
    df['fault_type'] = df[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1)
    
    return df

def preprocess_data(df, input_cols):
    """Preprocess data for unsupervised learning."""
    # Extract input features
    X = df[input_cols].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store fault types for evaluation
    y_true = None
    if 'fault_type' in df.columns:
        y_true = df['fault_type'].values
    
    return X_scaled, y_true

def perform_dimensionality_reduction(X, methods=None):
    """Apply dimensionality reduction techniques."""
    if methods is None:
        methods = ['pca', 'tsne', 'umap']
    
    results = {}
    
    if 'pca' in methods:
        print("\nPerforming PCA...")
        pca = PCA(n_components=2)
        results['pca'] = pca.fit_transform(X)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    if 'tsne' in methods:
        print("\nPerforming t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        results['tsne'] = tsne.fit_transform(X)
    
    if 'umap' in methods:
        print("\nPerforming UMAP...")
        reducer = umap.UMAP(random_state=42)
        results['umap'] = reducer.fit_transform(X)
    
    return results

def find_optimal_k(X, max_k=15):
    """Find optimal number of clusters using the elbow method and silhouette score."""
    print("\nFinding optimal number of clusters...")
    
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    db_scores = []  # Davies-Bouldin scores
    ch_scores = []  # Calinski-Harabasz scores
    
    for i in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Silhouette score (higher is better)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
        
        # Davies-Bouldin score (lower is better)
        db_scores.append(davies_bouldin_score(X, labels))
        
        # Calinski-Harabasz score (higher is better)
        ch_scores.append(calinski_harabasz_score(X, labels))
    
    # Plot evaluation metrics
    plt.figure(figsize=(20, 10))
    
    # WCSS (Elbow method)
    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    
    # Silhouette Score
    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score (higher is better)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    
    # Davies-Bouldin Score
    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_k + 1), db_scores, marker='o')
    plt.title('Davies-Bouldin Score (lower is better)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin Score')
    
    # Calinski-Harabasz Score
    plt.subplot(2, 2, 4)
    plt.plot(range(2, max_k + 1), ch_scores, marker='o')
    plt.title('Calinski-Harabasz Score (higher is better)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz Score')
    
    plt.tight_layout()
    plt.savefig("cluster_evaluation_metrics.png")
    plt.close()
    
    # Find optimal k
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2
    optimal_k_db = np.argmin(db_scores) + 2
    optimal_k_ch = np.argmax(ch_scores) + 2
    
    print(f"Optimal k according to silhouette score: {optimal_k_silhouette}")
    print(f"Optimal k according to Davies-Bouldin score: {optimal_k_db}")
    print(f"Optimal k according to Calinski-Harabasz score: {optimal_k_ch}")
    
    # Return a suggested k (we could use voting or prioritize one method)
    return optimal_k_silhouette

def apply_clustering(X, y_true=None, n_clusters=None):
    """Apply various clustering algorithms and evaluate them."""
    if n_clusters is None:
        n_clusters = find_optimal_k(X)
    
    print(f"\nApplying clustering algorithms with {n_clusters} clusters...")
    
    clustering_results = {}
    
    # K-Means clustering
    print("\nApplying K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    clustering_results['KMeans'] = {'model': kmeans, 'labels': kmeans_labels}
    
    # Gaussian Mixture Model
    print("\nApplying Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    clustering_results['GMM'] = {'model': gmm, 'labels': gmm_labels}
    
    # DBSCAN - Density-Based Spatial Clustering
    print("\nApplying DBSCAN clustering...")
    # We need to find a good eps value (the maximum distance between samples)
    from sklearn.neighbors import NearestNeighbors
    
    # Use nearest neighbors to find a good eps value
    neighbors = NearestNeighbors(n_neighbors=min(20, len(X)))
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort and plot distances to find the elbow
    distances = np.sort(distances[:, -1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('Nearest Neighbor Distances')
    plt.ylabel('Distance')
    plt.xlabel('Points sorted by distance')
    plt.savefig("Machine Learning/Usuprivsed ML/UML/dbscan_distances.png")
    plt.close()
    
    # Choose an eps value at the point of maximum curvature
    # This is a simple heuristic - you might want to pick the value manually
    eps_candidate = distances[int(0.1 * len(distances))]  # Take value at 10% of sorted distances
    
    print(f"Selected eps for DBSCAN: {eps_candidate:.4f}")
    
    dbscan = DBSCAN(eps=eps_candidate, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    clustering_results['DBSCAN'] = {'model': dbscan, 'labels': dbscan_labels}
    
    # Agglomerative Hierarchical Clustering
    print("\nApplying Agglomerative Hierarchical Clustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X)
    clustering_results['Agglomerative'] = {'model': agg, 'labels': agg_labels}
    
    # Evaluate clustering results
    for name, result in clustering_results.items():
        labels = result['labels']
        
        # Skip evaluation if all points are assigned to the same cluster or as noise (-1)
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
            print(f"{name}: Unable to evaluate - insufficient clusters")
            continue
        
        try:
            silhouette = silhouette_score(X, labels)
            print(f"{name} Silhouette Score: {silhouette:.4f}")
            
            # If DBSCAN finds noise points (label -1), exclude them from DB and CH scores
            valid_mask = labels != -1
            if sum(valid_mask) > n_clusters:  # Ensure we have enough points for evaluation
                db = davies_bouldin_score(X[valid_mask], labels[valid_mask])
                ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
                print(f"{name} Davies-Bouldin Score: {db:.4f}")
                print(f"{name} Calinski-Harabasz Score: {ch:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
        
        # If we have true labels, evaluate cluster purity
        if y_true is not None:
            evaluate_cluster_purity(labels, y_true, name)
    
    return clustering_results

def evaluate_cluster_purity(labels, y_true, method_name):
    """Evaluate how well clusters align with true labels."""
    # Count occurrences of each true label in each cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = {}
        
        true_label = y_true[i]
        if true_label not in clusters[label]:
            clusters[label][true_label] = 0
        clusters[label][true_label] += 1
    
    total_points = len(labels)
    weighted_purity = 0
    
    print(f"\n{method_name} - Cluster composition:")
    for cluster_label, true_labels in clusters.items():
        if cluster_label == -1:
            print(f"  Noise points: {sum(true_labels.values())}")
            continue
            
        cluster_size = sum(true_labels.values())
        most_common_label = max(true_labels.items(), key=lambda x: x[1])
        purity = most_common_label[1] / cluster_size
        weighted_purity += purity * (cluster_size / total_points)
        
        print(f"  Cluster {cluster_label} (size: {cluster_size}):")
        print(f"    Most common true label: '{most_common_label[0]}' ({most_common_label[1]} points, {purity:.2%} of cluster)")
        print(f"    Label distribution: {true_labels}")
    
    print(f"  Weighted purity: {weighted_purity:.4f}")

def visualize_clusters(X_reduced, labels, method, true_labels=None):
    """Visualize clustering results using reduced dimensions."""
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Convert labels to categorical for consistent colors
    unique_labels = np.unique(labels)
    
    # Create a scatter plot
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', 
                          alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
    
    # Mark cluster centers if available (for K-Means)
    if hasattr(method, 'cluster_centers_'):
        centers = method.cluster_centers_
        if hasattr(method, 'transform'):  # For methods that can transform to the reduced space
            centers_reduced = method.transform(centers)
            plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], 
                       s=200, marker='X', c='red', alpha=0.8, edgecolors='black')
    
    plt.title(f'Cluster Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Add a legend
    handles, _ = scatter.legend_elements()
    legend_labels = [f'Cluster {i}' for i in unique_labels]
    if -1 in unique_labels:
        legend_labels[list(unique_labels).index(-1)] = 'Noise'
    plt.legend(handles, legend_labels, title="Clusters")
    
    plt.savefig(f"Machine Learning/Usuprivsed ML/UML/cluster_visualization_{type(method).__name__}.png")
    plt.close()
    
    # If we have true labels, create a visualization with true labels
    if true_labels is not None:
        plt.figure(figsize=(12, 10))
        
        unique_true_labels = np.unique(true_labels)
        
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=[list(unique_true_labels).index(l) for l in true_labels], 
                             cmap='tab10', alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        
        plt.title(f'True Label Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        handles, _ = scatter.legend_elements()
        plt.legend(handles, unique_true_labels, title="True Labels")
        
        plt.savefig(f"Machine Learning/Usuprivsed ML/UML/true_label_visualization.png")
        plt.close()

def analyze_feature_importance(X, clustering_results, input_cols):
    """Analyze which features are most important for clustering."""
    # For each clustering result, analyze feature importance
    for name, result in clustering_results.items():
        labels = result['labels']
        
        # Skip if we have noise points
        if -1 in labels:
            continue
        
        # Compute feature means for each cluster
        feature_means = pd.DataFrame({
            'cluster': labels,
            **{col: X[:, i] for i, col in enumerate(input_cols)}
        }).groupby('cluster').mean()
        
        # Compute feature variances for each cluster
        feature_vars = pd.DataFrame({
            'cluster': labels,
            **{col: X[:, i] for i, col in enumerate(input_cols)}
        }).groupby('cluster').var()
        
        # Plot heatmap of feature means by cluster
        plt.figure(figsize=(12, 8))
        sns.heatmap(feature_means, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'{name} - Feature Means by Cluster')
        plt.savefig(f"Machine Learning/Usuprivsed ML/UML/feature_means_{name}.png")
        plt.close()
        
        # Plot heatmap of feature variances by cluster
        plt.figure(figsize=(12, 8))
        sns.heatmap(feature_vars, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title(f'{name} - Feature Variances by Cluster')
        plt.savefig(f"Machine Learning/Usuprivsed ML/UML/feature_vars_{name}.png")
        plt.close()
        
        # Compute overall feature importance by cluster separation
        importance_scores = []
        
        for i, col in enumerate(input_cols):
            # Compute between-cluster variance for this feature
            between_var = np.var([feature_means.iloc[c, i] for c in range(len(feature_means))])
            
            # Compute within-cluster variance for this feature
            within_var = np.mean([feature_vars.iloc[c, i] for c in range(len(feature_vars))])
            
            # F-ratio: higher ratio means better separation
            if within_var > 0:
                f_ratio = between_var / within_var
            else:
                f_ratio = float('inf')  # Infinite separation (perfect)
            
            importance_scores.append((col, f_ratio))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{name} - Feature Importance:")
        for col, score in importance_scores:
            print(f"  {col}: {score:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=[score[0] for score in importance_scores], 
                   y=[score[1] for score in importance_scores])
        plt.title(f'{name} - Feature Importance Scores')
        plt.xticks(rotation=45)
        plt.ylabel('F-ratio (higher = more important)')
        plt.tight_layout()
        plt.savefig(f"Machine Learning/Usuprivsed ML/UML/feature_importance_{name}.png")
        plt.close()

def detect_anomalies(X, clustering_results):
    """Detect potential anomalies in the data."""
    for name, result in clustering_results.items():
        model = result['model']
        labels = result['labels']
        
        # Different methods for anomaly detection depending on algorithm
        if name == 'KMeans':
            # For K-Means, use distance to nearest centroid
            distances = model.transform(X)
            min_distances = np.min(distances, axis=1)
            
            # Points with large distances to their centroids are potential anomalies
            threshold = np.percentile(min_distances, 95)  # Top 5% as anomalies
            anomalies = min_distances > threshold
            
            print(f"\n{name} anomaly detection:")
            print(f"  Found {sum(anomalies)} potential anomalies (5% threshold)")
            
            # Plot distance distribution
            plt.figure(figsize=(10, 6))
            plt.hist(min_distances, bins=50)
            plt.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold:.2f})')
            plt.title(f'{name} - Distance to Nearest Centroid')
            plt.xlabel('Distance')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(f"Machine Learning/Usuprivsed ML/UML/anomaly_detection_{name}.png")
            plt.close()
            
        elif name == 'DBSCAN':
            # For DBSCAN, noise points are already identified as anomalies
            anomalies = labels == -1
            
            print(f"\n{name} anomaly detection:")
            print(f"  Found {sum(anomalies)} noise points (anomalies)")
            
        elif name == 'GMM':
            # For GMM, use negative log likelihood
            neg_log_likelihood = -model.score_samples(X)
            
            # Points with low likelihood are potential anomalies
            threshold = np.percentile(neg_log_likelihood, 95)  # Top 5% as anomalies
            anomalies = neg_log_likelihood > threshold
            
            print(f"\n{name} anomaly detection:")
            print(f"  Found {sum(anomalies)} potential anomalies (5% threshold)")
            
            # Plot likelihood distribution
            plt.figure(figsize=(10, 6))
            plt.hist(neg_log_likelihood, bins=50)
            plt.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold:.2f})')
            plt.title(f'{name} - Negative Log Likelihood')
            plt.xlabel('Negative Log Likelihood')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(f"Machine Learning/Usuprivsed ML/UML/anomaly_detection_{name}.png")
            plt.close()

def main():
    # Define column names
    input_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
    
    # Load data
    filepath = "classData.csv"  # Update with your file path
    df = load_data(filepath)
    
    # Preprocess data
    X, y_true = preprocess_data(df, input_cols)
    
    # Perform dimensionality reduction
    reduction_results = perform_dimensionality_reduction(X)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_k(X)
    
    # Apply clustering algorithms
    clustering_results = apply_clustering(X, y_true, optimal_k)
    
    # Visualize clusters
    for name, result in clustering_results.items():
        # Use PCA or t-SNE for visualization
        vis_data = reduction_results['pca']  # Could use 'tsne' or 'umap' too
        visualize_clusters(vis_data, result['labels'], result['model'], y_true)
    
    # Analyze feature importance
    analyze_feature_importance(X, clustering_results, input_cols)
    
    # Detect anomalies
    detect_anomalies(X, clustering_results)
    
    print("\nUnsupervised learning analysis complete. See generated images for visualizations.")

if __name__ == "__main__":
    main()