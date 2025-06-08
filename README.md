
# 🍷 Wine Quality Clustering Using DBSCAN

## 📌 Business Problem
Understanding patterns in wine characteristics can aid producers and distributors in categorizing wines by similarity, detecting anomalies, or identifying quality bands without human labeling. Clustering helps discover natural groupings based on physicochemical properties.

### Real-World Use Case:
This project applies **DBSCAN clustering** on the UCI Wine Quality dataset to:
- Group wines by physicochemical similarity
- Detect potential outliers (noise wines)
- Visualize clusters using PCA and t-SNE
- Measure clustering quality with silhouette and Davies-Bouldin scores

## 📂 Dataset
- **Source**: UCI Machine Learning Repository – Wine Quality Dataset
- **Features**:
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`
  - `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
- **Target (optional)**: `quality` (not used in clustering)
- **Rows**: ~1600 (red wine samples)

## ⚙️ Pipeline
1. Load and scale data (StandardScaler)
2. Apply DBSCAN clustering
3. Visualize clusters using:
   - PCA (Principal Component Analysis)
   - t-SNE (t-Distributed Stochastic Neighbor Embedding)
4. Evaluate with:
   - Silhouette Score
   - Davies-Bouldin Index
5. Streamlit dashboard for analysis

## 💻 Folder Structure
```
wine-quality-dbscan-project/
├── data/                   # Raw dataset
├── models/                 # Saved clustering model
├── output/                 # Clustered data, visualizations, metrics
├── scripts/                # Preprocessing, training, evaluation
├── notebooks/              # Jupyter notebook for exploration
├── streamlit_app/          # Streamlit dashboard
├── .github/workflows/      # GitHub CI (optional)
├── Dockerfile              # For container deployment
├── docker-compose.yml      # Docker compose (optional)
├── requirements.txt        # Dependencies
├── README.md               # This file
```

## 🚀 Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python scripts/train.py --eps 1.0 --min_samples 5
```

### Evaluate Model
```bash
python scripts/evaluate.py
```

### Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```

## 📊 Dashboard Features
- Cluster distribution bar chart
- PCA and t-SNE 2D scatterplots
- Clustering metrics with silhouette and DB index
- Interactive exploration of clusters

## 🐳 Docker Support
```bash
docker build -t wine-dbscan .
docker run -p 8501:8501 wine-dbscan
```

Or use Docker Compose:
```bash
docker-compose up --build
```

## 🧪 Explainability
- While clustering is unsupervised, you can interpret cluster features using PCA/t-SNE loadings or integrate SHAP with a classifier on pseudo-labels.

## 📄 License
MIT License
