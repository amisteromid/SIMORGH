import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
cluster_df = pd.read_csv('clustered_0.3tsv', sep='\t', header=None, names=['representative', 'individual'])

# Create clusters from tsv
clusters = {}
for _, row in cluster_df.iterrows():
    rep = row['representative']
    ind = row['individual']
    if rep not in clusters:
        clusters[rep] = set()
    clusters[rep].update([rep, ind]) 

# Split the training into train + valid and extend to all members of clusters
train_clusters, tmp_clusters = train_test_split(list(clusters.keys()), test_size=0.2, random_state=42)
valid_clusters, test_clusters = train_test_split(tmp_clusters, test_size=0.5, random_state=42)


train_entries = [item for rep in train_clusters for item in clusters[rep]]
valid_entries = [item for rep in valid_clusters for item in clusters[rep]]
test_entries = [item for rep in test_clusters for item in clusters[rep]]


# Save the results to files
with open('train.txt', 'w') as f:
    for item in train_entries:
        f.write(f"{item}\n")
        
with open('valid.txt', 'w') as f:
    for item in valid_entries:
        f.write(f"{item}\n")

with open('test.txt', 'w') as f:
    for item in test_entries:
        f.write(f"{item}\n")
