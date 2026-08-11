import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
cluster_df = pd.read_csv('train_n_apo_plinder_clustered.tsv', sep='\t', header=None, names=['representative', 'individual'])
apo_ids = set(np.genfromtxt('apo_structures.txt', dtype=str))
train_ids = set(np.genfromtxt('../train.txt', dtype=str))


# Create clusters from tsv
clusters = {}
for _, row in cluster_df.iterrows():
    rep = row['representative']
    ind = row['individual']
    if rep not in clusters:
        clusters[rep] = set()
    clusters[rep].update([rep, ind]) 


for_test=[]
for rep, members in clusters.items():
    if members.isdisjoint(train_ids) and not members.isdisjoint(apo_ids):
        for_test.append(rep)


# Save the results to files
with open('apo_structures_for_test.txt', 'w') as f:
    for item in for_test:
        f.write(f"{item}\n")
