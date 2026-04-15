#!/usr/bin/env python3
"""
Example script demonstrating text clustering on the MOLE dataset.
This is a simplified version for quick testing and demonstration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cluster_and_plot_text import TextClusterAnalyzer
import matplotlib.pyplot as plt

def quick_clustering_demo():
    """Run a quick clustering demonstration."""
    print("Starting quick clustering demo...")
    
    # Initialize analyzer
    analyzer = TextClusterAnalyzer(dataset_path='evaluated_gemini_only')
    
    try:
        # Load dataset
        analyzer.load_dataset()
        
        # Create embeddings (using fewer features for speed)
        analyzer.create_embeddings(method='tfidf', max_features=500)
        
        # Reduce dimensions
        analyzer.reduce_dimensions(method='svd', n_components=30)
        
        # Cluster texts (try with 4 clusters)
        silhouette_score, ch_score = analyzer.cluster_texts(method='kmeans', n_clusters=4)
        
        print(f"\nClustering Results:")
        print(f"- Number of documents: {len(analyzer.texts)}")
        print(f"- Number of clusters: {len(set(analyzer.cluster_labels))}")
        print(f"- Silhouette Score: {silhouette_score:.3f}")
        print(f"- Calinski-Harabasz Score: {ch_score:.3f}")
        
        # Show cluster distribution
        print("\nCluster Distribution:")
        cluster_counts = {}
        for label in analyzer.cluster_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        for cluster_id, count in sorted(cluster_counts.items()):
            percentage = (count / len(analyzer.cluster_labels)) * 100
            print(f"  Cluster {cluster_id}: {count} documents ({percentage:.1f}%)")
        
        # Get top keywords for each cluster
        print("\nTop Keywords per Cluster:")
        cluster_keywords = analyzer.analyze_cluster_keywords(top_n=5)
        
        # Create a simple visualization
        print("\nCreating visualizations...")
        analyzer.plot_cluster_distribution()
        
        return analyzer
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        print("Make sure the dataset 'evaluated_gemini_only' exists and contains text content.")
        return None

def compare_different_cluster_numbers():
    """Compare clustering performance with different numbers of clusters."""
    print("\nComparing different numbers of clusters...")
    
    analyzer = TextClusterAnalyzer(dataset_path='evaluated_gemini_only')
    
    try:
        # Load and process data
        analyzer.load_dataset()
        analyzer.create_embeddings(method='tfidf', max_features=500)
        analyzer.reduce_dimensions(method='svd', n_components=30)
        
        # Test different numbers of clusters
        cluster_range = range(2, 11)
        silhouette_scores = []
        ch_scores = []
        
        for n_clusters in cluster_range:
            print(f"Testing {n_clusters} clusters...")
            silhouette_score, ch_score = analyzer.cluster_texts(method='kmeans', n_clusters=n_clusters)
            silhouette_scores.append(silhouette_score)
            ch_scores.append(ch_score)
        
        # Plot the results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, ch_scores, 'ro-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal number of clusters based on silhouette score
        optimal_n = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        print(f"\nOptimal number of clusters based on silhouette score: {optimal_n}")
        
        # Run final clustering with optimal number
        analyzer.cluster_texts(method='kmeans', n_clusters=optimal_n)
        analyzer.plot_cluster_distribution()
        
        return analyzer, silhouette_scores, ch_scores
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return None, None, None

if __name__ == "__main__":
    print("Text Clustering Demo for MOLE Dataset")
    print("=" * 50)
    
    # Run quick demo
    analyzer = quick_clustering_demo()
    
    if analyzer is not None:
        # Ask user if they want to compare different cluster numbers
        response = input("\nWould you like to compare different numbers of clusters? (y/n): ")
        if response.lower() == 'y':
            compare_different_cluster_numbers()
    
    print("\nDemo complete!")
