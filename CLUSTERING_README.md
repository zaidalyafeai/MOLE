# Text Clustering and Visualization for MOLE Dataset

This directory contains tools for clustering and visualizing the text content of papers in the MOLE dataset.

## Files

- `src/cluster_and_plot_text.py` - Main clustering analysis class and functions
- `src/example_clustering.py` - Simple examples and demonstrations
- `CLUSTERING_README.md` - This documentation file

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

The clustering functionality requires these additional packages:
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plotting
- `nltk` - Natural language processing
- `datasets` - Dataset handling

## Quick Start

### Basic Clustering

Run the simple example to get started:

```bash
cd src
python example_clustering.py
```

This will:
1. Load the dataset from `evaluated_gemini_only`
2. Create TF-IDF embeddings
3. Reduce dimensions using SVD
4. Cluster using K-Means
5. Show cluster distribution and top keywords

### Full Analysis

For a comprehensive analysis with all visualizations and reports:

```python
from cluster_and_plot_text import TextClusterAnalyzer

# Initialize analyzer
analyzer = TextClusterAnalyzer(dataset_path='evaluated_gemini_only')

# Run complete analysis
results = analyzer.run_full_analysis(n_clusters=5, save_dir='clustering_results')
```

This will create a `clustering_results` directory containing:
- `cluster_distribution.png` - Cluster distribution plots
- `cluster_2d_plot.html` - Interactive 2D visualization
- `cluster_keywords.png` - Top keywords per cluster
- `clustering_report.txt` - Detailed analysis report

## Advanced Usage

### Custom Clustering Parameters

```python
# Load and process data
analyzer = TextClusterAnalyzer(dataset_path='evaluated_gemini_only')
analyzer.load_dataset()

# Create embeddings with custom parameters
analyzer.create_embeddings(
    method='tfidf',           # or 'count'
    max_features=2000,         # Number of features
    ngram_range=(1, 3),        # Use up to trigrams
    min_df=3,                  # Minimum document frequency
    max_df=0.7                 # Maximum document frequency
)

# Reduce dimensions
analyzer.reduce_dimensions(
    method='svd',              # or 'pca'
    n_components=100           # Number of components
)

# Cluster with different algorithms
analyzer.cluster_texts(
    method='kmeans',           # or 'dbscan', 'hierarchical'
    n_clusters=8,              # Number of clusters
    random_state=42
)
```

### Different Clustering Algorithms

#### K-Means
```python
analyzer.cluster_texts(
    method='kmeans',
    n_clusters=5,
    init='k-means++',
    n_init=10,
    max_iter=300
)
```

#### DBSCAN (Density-based)
```python
analyzer.cluster_texts(
    method='dbscan',
    eps=0.5,
    min_samples=5
)
```

#### Hierarchical Clustering
```python
analyzer.cluster_texts(
    method='hierarchical',
    n_clusters=5,
    linkage='ward'             # or 'complete', 'average', 'single'
)
```

### Custom Visualizations

```python
# Plot cluster distribution
analyzer.plot_cluster_distribution(save_path='my_clusters.png')

# Create 2D visualization
analyzer.plot_2d_visualization(save_path='my_2d_plot.html')

# Analyze and plot keywords
cluster_keywords = analyzer.analyze_cluster_keywords(top_n=15)
analyzer.plot_cluster_keywords(cluster_keywords, save_path='my_keywords.png')
```

## Understanding the Output

### Cluster Metrics

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better.
- **Calinski-Harabasz Score**: Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.

### Visualizations

1. **Cluster Distribution**: Shows the number of documents in each cluster
2. **2D Visualization**: Interactive scatter plot showing document relationships
3. **Keywords Plot**: Top terms that characterize each cluster
4. **Text Length Analysis**: Distribution of document lengths across clusters

### Cluster Keywords

The analysis identifies the most characteristic terms for each cluster using TF-IDF scores. These keywords help interpret what themes or topics each cluster represents.

## Troubleshooting

### Common Issues

1. **Dataset not found**: Make sure `evaluated_gemini_only` exists or specify the correct path
2. **Memory issues**: Reduce `max_features` or `n_components` for large datasets
3. **Poor clustering**: Try different numbers of clusters or preprocessing parameters

### Performance Tips

- For large datasets, start with fewer features (`max_features=500`)
- Use SVD instead of PCA for sparse matrices
- Experiment with different clustering algorithms
- Consider text preprocessing parameters (n-grams, stop words, etc.)

## Examples

### Finding Optimal Number of Clusters

```python
# Test different numbers of clusters
silhouette_scores = []
cluster_range = range(2, 15)

for n_clusters in cluster_range:
    analyzer.cluster_texts(method='kmeans', n_clusters=n_clusters)
    score = silhouette_score(analyzer.reduced_embeddings, analyzer.cluster_labels)
    silhouette_scores.append(score)

# Plot results
import matplotlib.pyplot as plt
plt.plot(cluster_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
```

### Custom Text Preprocessing

```python
def custom_preprocess(texts):
    # Add your custom preprocessing here
    processed = []
    for text in texts:
        # Your preprocessing logic
        processed.append(text)
    return processed

# Use with the analyzer
analyzer.preprocess_texts = custom_preprocess
```

## Integration with Existing Code

The clustering tools are designed to work with the existing MOLE dataset structure. They use the same dataset loading and text extraction methods as `visualize_dataset.py`.

You can easily integrate clustering results into your existing workflows:

```python
# After clustering, add cluster labels to your dataset
dataset_with_clusters = analyzer.dataset.add_column('cluster', analyzer.cluster_labels)

# Save the enhanced dataset
dataset_with_clusters.save_to_disk('dataset_with_clusters')
```

## Future Enhancements

Potential improvements to consider:
- Word embeddings (Word2Vec, GloVe, FastText)
- Transformer-based embeddings (BERT, Sentence-BERT)
- Topic modeling (LDA, NMF)
- Semantic search capabilities
- Real-time clustering of new documents
