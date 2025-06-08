
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Comparison - Wine Dataset", layout="wide")
st.title("🔍 Clustering Comparison Dashboard")

df = pd.read_csv("output/clustered_data.csv")

tabs = st.tabs(["📊 Distribution", "🧬 PCA View", "🔮 t-SNE View", "📈 Metrics"])

with tabs[0]:
    st.header("Cluster Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='cluster', ax=ax1)
    st.pyplot(fig1)

with tabs[1]:
    st.header("PCA Projection")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', ax=ax2)
    st.pyplot(fig2)

with tabs[2]:
    st.header("t-SNE Projection")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster', palette='tab10', ax=ax3)
    st.pyplot(fig3)

with tabs[3]:
    st.header("Clustering Metrics")
    with open("output/cluster_metrics.txt") as f:
        metrics = f.read()
    st.code(metrics)
