
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

def evaluate_clusters(file_path='output/clustered_data.csv'):
    df = pd.read_csv(file_path)
    if 'cluster' not in df.columns:
        print("No cluster column found.")
        return

    features = df.drop(['quality', 'cluster'], axis=1)

    # Plot cluster distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='cluster')
    plt.title("Cluster Distribution")
    plt.savefig("output/cluster_distribution.png")

    # PCA plot
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(features)
    df['pca1'], df['pca2'] = pca_components[:, 0], pca_components[:, 1]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
    plt.title("PCA - Cluster View")
    plt.savefig("output/pca_clusters.png")

    # TSNE plot
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_components = tsne.fit_transform(features)
    df['tsne1'], df['tsne2'] = tsne_components[:, 0], tsne_components[:, 1]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', palette='tab10')
    plt.title("t-SNE - Cluster View")
    plt.savefig("output/tsne_clusters.png")

    # Metrics
    mask = df['cluster'] != -1  # ignore noise if any
    silhouette = silhouette_score(features[mask], df['cluster'][mask]) if mask.sum() > 1 else -1
    db_score = davies_bouldin_score(features[mask], df['cluster'][mask]) if mask.sum() > 1 else -1

    with open("output/cluster_metrics.txt", "w") as f:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Davies-Bouldin Index: {db_score:.4f}\n")

    print("Evaluation complete. Metrics and visuals saved in output/")
