import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import joblib

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Select numeric columns only
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Drop missing values
numeric_df.dropna(inplace=True)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Train Hierarchical (Agglomerative) Clustering
model = AgglomerativeClustering(
    n_clusters=3,     # can be changed later
    linkage="ward"
)

model.fit(scaled_data)

# Save scaler and features
joblib.dump(
    {
        "scaler": scaler,
        "features": numeric_df.columns.tolist(),
        "labels": model.labels_
    },
    "model.pkl"
)

print("✅ Hierarchical clustering training completed")