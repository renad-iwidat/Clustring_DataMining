# Driving Behavior Clustering

This project analyzes driving behaviors using clustering techniques to identify patterns based on average daily distance traveled and the percentage of time spent over the speed limit. By understanding these patterns, we can classify drivers into groups that reflect different driving habits.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Steps](#steps)
  - [1. Load the Dataset](#1-load-the-dataset)
  - [2. Visualize the Data](#2-visualize-the-data)
  - [3. Feature Scaling](#3-feature-scaling)
  - [4. Apply K-Means Clustering](#4-apply-k-means-clustering)
  - [5. Visualize Clusters](#5-visualize-clusters)
  - [6. Interpret the Clusters](#6-interpret-the-clusters)
- [Run the Code](#run-the-code)
- [Conclusion](#conclusion)

---

## Overview

The objective of this project is to group driving records based on their characteristics to understand different driving patterns:

- **Features**: Average daily distance traveled (`mean_dist_day`) and the percentage of time spent over the speed limit (`mean_over_speed_perc`).
- **Methodology**: Clustering with the k-means algorithm.

## Dataset

The dataset, `driver-data.csv`, contains anonymized driving records. The key features include:

- `mean_dist_day`: Average distance traveled per day.
- `mean_over_speed_perc`: Percentage of time spent exceeding the speed limit.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Steps

### 1. Load the Dataset

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('driver-data.csv')
df.head()
```

### 2. Visualize the Data

```python
# Plotting the original data
plt.figure(figsize=(10, 6))
plt.scatter(df['mean_dist_day'], df['mean_over_speed_perc'], c='blue', s=50)
plt.xlabel('Mean Distance per Day')
plt.ylabel('Mean Over Speed Percentage')
plt.title('Mean Distance per Day vs Mean Over Speed Percentage')
plt.show()
```

### 3. Feature Scaling

```python
# Extract features for clustering
X = df[['mean_dist_day', 'mean_over_speed_perc']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Apply K-Means Clustering

```python
# Apply k-means clustering
kmeans = KMeans(n_clusters=6, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

### 5. Visualize Clusters

```python
# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['mean_dist_day'], df['mean_over_speed_perc'], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Mean Distance per Day')
plt.ylabel('Mean Over Speed Percentage')
plt.title('Clusters of Mean Distance per Day and Mean Over Speed Percentage')

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()
```

### 6. Interpret the Clusters

- **Cluster 0** (Low Distance, Low Speeding): Likely cautious drivers who travel short distances.
- **Cluster 1** (Moderate Distance, Moderate Speeding): Moderate daily travel and occasional speeding.
- **Cluster 2** (Moderate Distance, High Speeding): Frequent speeders with moderate daily travel.
- **Cluster 3** (High Distance, Moderate Speeding): Long-distance drivers with occasional speeding.
- **Cluster 4** (High Distance, Low Speeding): Long-distance drivers who rarely speed.
- **Cluster 5** (Very High Distance, High Speeding): Heavy usage with frequent speeding.

## Run the Code

Save the code in a Python script (e.g., `clustering_analysis.py`) and run it:

```bash
python clustering_analysis.py
```

## Conclusion

This project demonstrates how clustering can identify distinct driving behaviors. The insights gained can be used for safety initiatives, targeted interventions, or understanding usage patterns in fleet management systems.

---
