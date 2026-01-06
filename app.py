import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hierarchical Clustering", layout="centered")
st.title("🌳 Hierarchical Clustering Visualization")

# Load data
df = pd.read_csv("data/dataset.csv")
numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna()

# Load saved objects
bundle = joblib.load("model.pkl")
scaler = bundle["scaler"]
features = bundle["features"]
labels = bundle["labels"]

# Scale data
scaled_data = scaler.transform(numeric_df)

st.subheader("📌 Dendrogram")

linked = linkage(scaled_data, method="ward")

plt.figure(figsize=(8, 4))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Hierarchical Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
st.pyplot(plt)

st.subheader("📊 Cluster Visualization (First 2 Features)")

plt.figure(figsize=(6, 4))
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=labels,
    cmap="viridis"
)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Hierarchical Clusters")
st.pyplot(plt)

