import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
cluster_df = pd.read_csv('cl1_cluster.tsv', sep='\t', header=None, names=['representative', 'individual']) # train (BY CHAIN) + crypto 
sel = np.genfromtxt('test_ids.txt', dtype=np.dtype('U')) # test subset of crypto
sel_set = set(sel)
ids = np.genfromtxt('all_ids.txt', dtype=np.dtype('U')) # crypto + train

# Create clusters from tsv
clusters = {}
for _, row in cluster_df.iterrows():
    rep = row['representative']
    ind = row['individual']
    if rep not in clusters:
        clusters[rep] = set()
    clusters[rep].update([rep, ind])

train_chains = []
valid_chains = []
for rep, entries in clusters.items():
    if sel_set.intersection(entries):
        valid_chains.extend(entries)
    else:
        train_chains.extend(entries)

valid_chain_pdbs = set([c[:4] for c in valid_chains])
train_chain_pdbs = set([c[:4] for c in train_chains])

train_set = []
valid_set = []
for i in ids:
    if i in valid_chains or i in valid_chain_pdbs:
        valid_set.append(i)
    elif i in train_chains or i in train_chain_pdbs:
        train_set.append(i)
    else:
        print ('NEITHER !!!!!!!!')
        
#print (len(valid_set))
valid_set = [i for i in valid_set if (i not in sel) and (i not in [i[:4] for i in sel])]
#print (len(valid_set))
#print (len(train_set))

with open('train.txt', 'w') as f:
    f.write('\n'.join(train_set))
with open('valid.txt', 'w') as f:
    f.write('\n'.join(valid_set))
