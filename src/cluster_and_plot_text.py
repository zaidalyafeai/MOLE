#!/usr/bin/env python3
"""
Cluster and plot text content from the MOLE dataset.
This script performs text clustering using various embedding and clustering methods,
then creates visualizations to explore the dataset structure.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Dataset handling
from datasets import load_from_disk

# ML and clustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Text processing
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Local imports
from utils import create_hash

class TextClusterAnalyzer:
    """Analyze and cluster text content from the MOLE dataset."""
    
    def __init__(self, dataset_path: str = 'evaluated_gemini_only'):
        """Initialize the analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.dataset = None
        self.texts = []
        self.embeddings = None
        self.cluster_labels = None
        self.reduced_embeddings = None
        
    def load_dataset(self):
        """Load the dataset and extract text content."""
        print("Loading dataset...")
        self.dataset = load_from_disk(self.dataset_path)
        
        # Add text content if not already present
        if 'text' not in self.dataset.column_names:
            self.dataset = self.dataset.map(self._add_text)
        
        # Extract texts
        self.texts = [example['text'] for example in self.dataset]
        print(f"Loaded {len(self.texts)} documents")
        
    def _add_text(self, example):
        """Add text content to dataset examples."""
        try:
            link = json.load(open(example['path']))['config']['link']
            paper_text = open(f'static/papers/{create_hash(link)}/paper_text.txt').read()
            return {'text': paper_text}
        except Exception as e:
            print(f"Error loading text for {example['path']}: {e}")
            return {'text': ''}
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for analysis."""
        processed_texts = []
        stop_words = set(stopwords.words('english'))
        
        for text in texts:
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = text.lower().strip()
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            processed_texts.append(' '.join(tokens))
        
        return processed_texts
    
    def create_embeddings(self, method: str = 'tfidf', max_features: int = 1000):
        """Create text embeddings using specified method."""
        print(f"Creating embeddings using {method}...")
        
        processed_texts = self.preprocess_texts(self.texts)
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        self.embeddings = vectorizer.fit_transform(processed_texts)
        self.vectorizer = vectorizer
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        
    def reduce_dimensions(self, method: str = 'svd', n_components: int = 50):
        """Reduce dimensionality of embeddings."""
        print(f"Reducing dimensions using {method}...")
        
        if method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        if hasattr(self.embeddings, 'toarray'):
            embeddings_dense = self.embeddings.toarray()
        else:
            embeddings_dense = self.embeddings
            
        self.reduced_embeddings = reducer.fit_transform(embeddings_dense)
        self.reducer = reducer
        print(f"Reduced to {n_components} dimensions")
        
    def cluster_texts(self, method: str = 'kmeans', n_clusters: int = 5, **kwargs):
        """Cluster texts using specified method."""
        print(f"Clustering using {method}...")
        
        if self.reduced_embeddings is None:
            raise ValueError("Must call reduce_dimensions() first")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        elif method == 'dbscan':
            clusterer = DBSCAN(**kwargs)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
        self.clusterer = clusterer
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(self.reduced_embeddings, self.cluster_labels)
        calinski_harabasz_avg = calinski_harabasz_score(self.reduced_embeddings, self.cluster_labels)
        
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.3f}")
        print(f"Found {len(set(self.cluster_labels))} clusters")
        
        return silhouette_avg, calinski_harabasz_avg
    
    def plot_cluster_distribution(self, save_path: str = None):
        """Plot cluster distribution."""
        plt.figure(figsize=(12, 8))
        
        # Cluster distribution
        plt.subplot(2, 2, 1)
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        cluster_counts.plot(kind='bar')
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=0)
        
        # Cluster sizes pie chart
        plt.subplot(2, 2, 2)
        plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Cluster Proportions')
        
        # Text length distribution by cluster
        plt.subplot(2, 2, 3)
        text_lengths = [len(text.split()) for text in self.texts]
        df_viz = pd.DataFrame({
            'cluster': self.cluster_labels,
            'text_length': text_lengths
        })
        sns.boxplot(data=df_viz, x='cluster', y='text_length')
        plt.title('Text Length by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Word Count')
        
        # Cluster quality metrics
        plt.subplot(2, 2, 4)
        silhouette_scores = []
        for i in range(2, min(11, len(set(self.cluster_labels)) + 1)):
            kmeans_temp = KMeans(n_clusters=i, random_state=42)
            labels_temp = kmeans_temp.fit_predict(self.reduced_embeddings)
            silhouette_scores.append(silhouette_score(self.reduced_embeddings, labels_temp))
        
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'bo-')
        plt.axvline(x=len(set(self.cluster_labels)), color='r', linestyle='--', alpha=0.7)
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_2d_visualization(self, save_path: str = None):
        """Create 2D visualization of clusters."""
        # Reduce to 2D for visualization
        svd_2d = TruncatedSVD(n_components=2, random_state=42)
        embeddings_2d = svd_2d.fit_transform(self.embeddings)
        
        # Create interactive plot
        fig = px.scatter(
            x=embeddings_2d[:, 0], 
            y=embeddings_2d[:, 1],
            color=self.cluster_labels,
            hover_data={'text': [text[:100] + '...' for text in self.texts]},
            title='2D Visualization of Text Clusters',
            labels={'x': 'SVD Component 1', 'y': 'SVD Component 2', 'color': 'Cluster'}
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        # Also create static matplotlib version
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('SVD Component 1')
        plt.ylabel('SVD Component 2')
        plt.title('2D Visualization of Text Clusters')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_cluster_keywords(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Analyze top keywords for each cluster."""
        print("Analyzing cluster keywords...")
        
        cluster_keywords = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in set(self.cluster_labels):
            # Get texts in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = self.embeddings[cluster_mask]
            
            # Calculate mean TF-IDF scores for this cluster
            mean_scores = np.mean(cluster_embeddings.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(mean_scores)[-top_n:][::-1]
            top_keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            cluster_keywords[cluster_id] = top_keywords
            
            print(f"\nCluster {cluster_id} Top Keywords:")
            for keyword, score in top_keywords:
                print(f"  {keyword}: {score:.4f}")
        
        return cluster_keywords
    
    def plot_cluster_keywords(self, cluster_keywords: Dict[int, List[Tuple[str, float]]], 
                            save_path: str = None):
        """Plot top keywords for each cluster."""
        n_clusters = len(cluster_keywords)
        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters))
        
        if n_clusters == 1:
            axes = [axes]
        
        for idx, (cluster_id, keywords) in enumerate(cluster_keywords.items()):
            words = [kw[0] for kw in keywords]
            scores = [kw[1] for kw in keywords]
            
            axes[idx].barh(words, scores)
            axes[idx].set_title(f'Cluster {cluster_id} - Top Keywords')
            axes[idx].set_xlabel('Mean TF-IDF Score')
            axes[idx].set_ylabel('Keywords')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path: str = 'clustering_report.txt'):
        """Generate a comprehensive clustering report."""
        report = []
        report.append("TEXT CLUSTERING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Dataset: {self.dataset_path}")
        report.append(f"Number of documents: {len(self.texts)}")
        report.append(f"Embedding method: TF-IDF")
        report.append(f"Dimensionality reduction: SVD")
        report.append(f"Clustering method: K-Means")
        report.append(f"Number of clusters: {len(set(self.cluster_labels))}")
        report.append("")
        
        # Cluster statistics
        report.append("CLUSTER STATISTICS")
        report.append("-" * 20)
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.texts)) * 100
            report.append(f"Cluster {cluster_id}: {count} documents ({percentage:.1f}%)")
        report.append("")
        
        # Text length statistics
        report.append("TEXT LENGTH STATISTICS")
        report.append("-" * 25)
        text_lengths = [len(text.split()) for text in self.texts]
        report.append(f"Average text length: {np.mean(text_lengths):.1f} words")
        report.append(f"Median text length: {np.median(text_lengths):.1f} words")
        report.append(f"Min text length: {np.min(text_lengths)} words")
        report.append(f"Max text length: {np.max(text_lengths)} words")
        report.append("")
        
        # Keywords for each cluster
        cluster_keywords = self.analyze_cluster_keywords(top_n=5)
        report.append("TOP KEYWORDS PER CLUSTER")
        report.append("-" * 30)
        for cluster_id, keywords in cluster_keywords.items():
            report.append(f"Cluster {cluster_id}:")
            for keyword, score in keywords:
                report.append(f"  - {keyword} ({score:.4f})")
            report.append("")
        
        # Save report
        report_text = '\n'.join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {save_path}")
        return report_text
    
    def run_full_analysis(self, n_clusters: int = 5, save_dir: str = 'clustering_results'):
        """Run complete clustering analysis pipeline."""
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("Starting full text clustering analysis...")
        
        # Load and process data
        self.load_dataset()
        self.create_embeddings(method='tfidf', max_features=1000)
        self.reduce_dimensions(method='svd', n_components=50)
        
        # Cluster texts
        silhouette_score, ch_score = self.cluster_texts(method='kmeans', n_clusters=n_clusters)
        
        # Generate visualizations
        print("Generating visualizations...")
        self.plot_cluster_distribution(save_path=save_dir / 'cluster_distribution.png')
        self.plot_2d_visualization(save_path=save_dir / 'cluster_2d_plot.html')
        
        # Analyze keywords
        cluster_keywords = self.analyze_cluster_keywords(top_n=10)
        self.plot_cluster_keywords(cluster_keywords, save_path=save_dir / 'cluster_keywords.png')
        
        # Generate report
        report = self.generate_report(save_path=save_dir / 'clustering_report.txt')
        
        print(f"\nAnalysis complete! Results saved to {save_dir}")
        print(f"Silhouette Score: {silhouette_score:.3f}")
        print(f"Calinski-Harabasz Score: {ch_score:.3f}")
        
        return {
            'silhouette_score': silhouette_score,
            'calinski_harabasz_score': ch_score,
            'cluster_keywords': cluster_keywords,
            'report': report
        }


def main():
    """Main function to run the clustering analysis."""
    # Initialize analyzer
    analyzer = TextClusterAnalyzer(dataset_path='evaluated_gemini_only')
    
    # Run full analysis
    results = analyzer.run_full_analysis(n_clusters=5, save_dir='clustering_results')
    
    print("\nClustering Analysis Summary:")
    print(f"- Found {len(set(analyzer.cluster_labels))} clusters")
    print(f"- Silhouette Score: {results['silhouette_score']:.3f}")
    print(f"- Calinski-Harabasz Score: {results['calinski_harabasz_score']:.3f}")
    print("- Visualizations and report saved to 'clustering_results' directory")


if __name__ == "__main__":
    main()
