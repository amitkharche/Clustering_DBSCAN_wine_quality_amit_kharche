
import argparse
import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scripts.preprocess import load_and_scale_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def cluster_and_save(data_path, model_path, eps=1.0, min_samples=5, output_path="output"):
    logging.info("Loading and scaling data...")
    X_scaled, df_original = load_and_scale_data(data_path)

    logging.info(f"Clustering using DBSCAN with eps={eps} and min_samples={min_samples}")
    db_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db_model.fit_predict(X_scaled)

    df_clustered = df_original.copy()
    df_clustered['cluster'] = cluster_labels

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(db_model, model_path)
    logging.info(f"Model saved at: {model_path}")

    # Save clustered data
    os.makedirs(output_path, exist_ok=True)
    clustered_path = os.path.join(output_path, "clustered_data.csv")
    df_clustered.to_csv(clustered_path, index=False)
    logging.info(f"Clustered data saved to: {clustered_path}")

    # Save metrics inline
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logging.info(f"Total clusters found (excluding noise): {n_clusters}")
    logging.info(f"Noise points: {(cluster_labels == -1).sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DBSCAN clustering model")
    parser.add_argument("--data_path", type=str, default="data/winequality.csv", help="Path to input data")
    parser.add_argument("--model_path", type=str, default="models/dbscan_model.pkl", help="Path to save DBSCAN model")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save clustered output")
    parser.add_argument("--eps", type=float, default=1.0, help="Epsilon parameter for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Min samples for DBSCAN")

    args = parser.parse_args()

    cluster_and_save(
        data_path=args.data_path,
        model_path=args.model_path,
        eps=args.eps,
        min_samples=args.min_samples,
        output_path=args.output_path
    )
