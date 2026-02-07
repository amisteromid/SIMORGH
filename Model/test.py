from tqdm import tqdm
import torch
import h5py
from sys import argv
import numpy as np
from architecture.model_gnn import Model
from architecture.model_set import SetModel
from architecture.config import config_model, config_runtime
from utils_test import Dataset, collate_set_transformer, setup_dataloader
from scoring import bc_scoring, bc_score_names, nanmean
from hdbscan import HDBSCAN
from sklearn.metrics import pairwise_distances





model1_num = argv[1]
model2_num = argv[2]
config_data = {
    'dataset_filepath': "/home/omokhtari/deliqate/Data/ds3/db_bbflow.h5",
    'raw_dataset_filepath': '/home/omokhtari/deliqate/Data/ds3/coords_labels.h5',
    'test_selection_filepath': '/home/omokhtari/deliqate/Data/p2rank/splitting/valid.txt'}

def eval_step(model1, model2, device, batch_data):
    # get embs
    seq, SCOV, nn_ids, D, R, SCOD, y, protein_id  = [data for data in batch_data]
    emb_list=[]
    for i in range(len(SCOV)):
        num_nodes, k = nn_ids[i].shape
        edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
        edge_dst = nn_ids[i].flatten()
        with torch.no_grad():
            emb = model1([[seq.to(device), SCOV[i].to(device)], [SCOD[i].to(device), R[i].to(device), D[i].to(device)]],edge_src.to(device),edge_dst.to(device),get_mor=True)
        emb_list.append(emb)
    emb_list = torch.stack(emb_list, dim=1)
    # evalluate with setmodeli
    z = model2.forward(emb_list)
    return z.to(device), y.to(device), protein_id



from sklearn.cluster import MeanShift, AgglomerativeClustering
from scipy.spatial.distance import cdist
def calculate_dca_dcc_robust(coords, predictions, ground_truth, threshold=0.5, pocket_radius=8.0):
    # --- 1. Data Cleaning & Shape Handling ---
    if hasattr(coords, 'cpu'): coords = coords.cpu().numpy()
    if hasattr(predictions, 'cpu'): predictions = predictions.cpu().numpy()
    if hasattr(ground_truth, 'cpu'): ground_truth = ground_truth.cpu().numpy()
    
    coords = np.array(coords)
    predictions = np.array(predictions).flatten()
    ground_truth = np.array(ground_truth).flatten()

    # Use Alpha Carbon (index 1 usually, or 0 depending on pdb parsing) 
    # ensuring we have (N, 3) shape
    if coords.ndim == 3:
        coords = coords[:, 1, :] if coords.shape[1] > 1 else coords[:, 0, :]

    # --- 2. Ground Truth Processing ---
    gt_mask = ground_truth == 1
    if not gt_mask.any():
        # Edge case: Protein has no binding site defined in GT
        return float('inf'), float('inf')
        
    gt_coords = coords[gt_mask]
    # GT Center: Geometric center of all binding residues
    gt_center = gt_coords.mean(axis=0)

    # --- 3. Prediction Filtering (Fixed Threshold) ---
    # We only cluster residues the model is actually confident about
    pred_mask = predictions > threshold
    
    # Fallback: If no predictions > threshold, take top 3 residues 
    # (prevents NaNs if model is weak but correctly ranks residues)
    if not pred_mask.any():
        top_k_indices = np.argsort(predictions)[-3:]
        pred_mask = np.zeros_like(predictions, dtype=bool)
        pred_mask[top_k_indices] = True
        # If max probability is really low (e.g. < 0.1), might just return inf
        if predictions[top_k_indices].max() < 0.1:
             return float('inf'), float('inf')

    pred_coords = coords[pred_mask]
    pred_probs = predictions[pred_mask]

    # --- 4. Clustering (MeanShift) ---
    # MeanShift is great for binding sites because it finds the "mode" (dense center)
    # bandwidth determines the radius of the pocket (typically 5-10 Angstroms)
    try:
        ms = MeanShift(bandwidth=pocket_radius, bin_seeding=True)
        labels = ms.fit_predict(pred_coords)
        cluster_ids = np.unique(labels)
    except ValueError:
        # Fallback for extremely edge cases (e.g. 1 point)
        cluster_ids = [0]
        labels = np.zeros(len(pred_coords))

    # --- 5. Weighted Center Calculation ---
    predicted_centers = []
    
    for cid in cluster_ids:
        # Get points belonging to this cluster
        cluster_mask = labels == cid
        c_coords = pred_coords[cluster_mask]
        c_probs = pred_probs[cluster_mask]
        
        # WEIGHTED MEAN: High probability residues pull the center closer to them
        # Center = sum(coord * prob) / sum(prob)
        weights = c_probs[:, np.newaxis]
        weighted_center = np.sum(c_coords * weights, axis=0) / np.sum(weights)
        predicted_centers.append(weighted_center)
    
    predicted_centers = np.array(predicted_centers)

    # --- 6. Metric Calculation ---
    
    # DCA: Distance to Closest Atom
    # Calculate dists between all predicted centers and all GT atoms
    # cdist returns matrix (n_centers x n_gt_atoms)
    dists_to_atoms = cdist(predicted_centers, gt_coords)
    
    # Min distance from any center to any GT atom
    dca = np.min(dists_to_atoms)

    # DCC: Distance to Closest Center
    # Distance from any predicted center to the single GT center
    dists_to_center = np.linalg.norm(predicted_centers - gt_center, axis=1)
    dcc = np.min(dists_to_center)

    return dca, dcc




def calculate_dca_dcc(coords, predictions, ground_truth, min_samples=5, alpha=2):
    
    # Convert to numpy and handle shapes
    if hasattr(coords, 'cpu'):  # PyTorch tensor
        coords = coords.cpu().numpy()
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu().numpy()
    
    coords = np.array(coords)
    predictions = np.array(predictions).flatten()
    ground_truth = np.array(ground_truth).flatten()
    
    # Handle (N, 2, 3) shape - use CA atoms (index 0)
    if coords.ndim == 3:
        coords = coords[:, 0, :]
    
    # Get ground truth coordinates and center
    gt_mask = ground_truth == 1
    if not gt_mask.any():
        #print("RETURN NONE: No ground truth binding sites (all zeros)")
        return None, None
    
    gt_coords = coords[gt_mask]
    gt_center = gt_coords.mean(axis=0)
    #print(f"Ground truth sites: {gt_mask.sum()}")
    
    # Filter by median prediction threshold
    median_pred = np.median(predictions)
    pred_mask = predictions > 0.3 #median_pred
    
    if not pred_mask.any():
        #print(f"RETURN NONE: No predictions above median threshold ({median_pred:.4f})")
        return None, None
    
    pred_coords = coords[pred_mask]
    pred_probs = predictions[pred_mask]
    #print(f"Predictions above median: {pred_mask.sum()}/{len(coords)} (median: {median_pred:.4f})")
    
    # Custom distance metric
    def weighted_metric(p, q):
        euclidean = np.linalg.norm(p[:-1] - q[:-1])
        prob_weight = alpha / (p[-1] * q[-1])
        return euclidean #+ prob_weight
    
    # Cluster with HDBSCAN
    coords_with_probs = np.hstack((pred_coords, pred_probs.reshape(-1, 1)))
    dist_matrix = pairwise_distances(coords_with_probs, metric=weighted_metric)
    
    clusterer = HDBSCAN(min_cluster_size=min_samples, metric='precomputed')
    labels = clusterer.fit_predict(dist_matrix)
    
    # Get valid clusters (exclude noise label -1)
    valid_mask = labels != -1
    if not valid_mask.any():
        #print(f"RETURN NONE: HDBSCAN found no valid clusters (all noise, min_samples={min_samples})")
        #print(f"  Total points clustered: {len(labels)}, All labeled as noise: {(labels == -1).sum()}")
        return None, None
    
    valid_labels = np.unique(labels[valid_mask])
    #print(f"Valid clusters found: {len(valid_labels)}")
    
    # Calculate cluster centers
    cluster_centers = np.array([
        pred_coords[labels == label].mean(axis=0) 
        for label in valid_labels
    ])
    
    # DCA: minimum distance from any cluster center to any GT atom
    dca_distances = np.linalg.norm(
        gt_coords[:, np.newaxis] - cluster_centers, axis=2
    )
    min_dca = dca_distances.min()
    
    # DCC: minimum distance from any cluster center to GT center
    dcc_distances = np.linalg.norm(cluster_centers - gt_center, axis=1)
    min_dcc = dcc_distances.min()
    
    #print(f"SUCCESS: DCA={min_dca:.3f}, DCC={min_dcc:.3f}")
    return min_dca, min_dcc

def test(config_data, config_model, config_runtime, output_file="test_predictions.txt"):
    device = torch.device(config_runtime['device'])
    db_coords = h5py.File(config_data['raw_dataset_filepath'], 'r')
    
    model1 = Model(config_model).to(device)
    ckpt1 = torch.load(f"model_{model1_num}.pt", map_location=device, weights_only=True)
    model1.load_state_dict(ckpt1["model_state_dict"] if "model_state_dict" in ckpt1 else ckpt1)
    
    model2 = SetModel(config_model).to(device)
    ckpt2 = torch.load(f"model_{model2_num}.pt", map_location=device, weights_only=True)
    model2.load_state_dict(ckpt2["model_state_dict"] if "model_state_dict" in ckpt2 else ckpt2)
    
    test_ids = np.genfromtxt(config_data['test_selection_filepath'], dtype=np.dtype('U'))
    dataloader_test = setup_dataloader(config_data, test_ids)
    model1.eval()
    model2.eval()


    with torch.no_grad(), h5py.File(output_file, "w") as f:
        scores_list=[]
        print (len(dataloader_test))
        for batch_data in tqdm(dataloader_test):
            z, y, protein_id = eval_step(model1, model2, device, batch_data)
            #print(f"Percentage of 1s: {y.float().mean().item():.4f}")
            p = torch.sigmoid(z)
            # get DCC DCA
            seq = db_coords[protein_id]['sequence'][()].decode('utf-8')

            # create group per protein
            grp = f.create_group(str(protein_id))
            grp.create_dataset("sequence", data=seq)
            grp.create_dataset("pp", data=p.detach().cpu().numpy())

            #coords = db_coords[protein_id]['coordinates'][()][0]
            #dca, dcc = calculate_dca_dcc_robust(coords, p, y)
            #if dca is None or dcc is None:
            #    dca, dcc = 10.0, 10.0
            #print (dca, dcc)
            #scores = bc_scoring(y, p, cutoff=0.45)
            #scores = torch.cat([scores, torch.tensor([[dcc],[dca]],device=scores.device, dtype=scores.dtype)], dim=0)
            #scores_list.append(scores)

        #scores = torch.stack(scores_list, dim=0)
        #nan_counts = torch.isnan(scores).sum(dim=0)
        #means = nanmean(scores)
        #score_names = ['precision','recall','mcc','PR','F1', 'dcc','dca']
        #for i, name in enumerate(score_names):
        #    print(f"{name}: mean={means[i].item():.4f}, NaNs={nan_counts[i].item()}")
        #dcc_success = (scores[:, 5] < 4).sum().item() / scores.shape[0] * 100
        #dca_success = (scores[:, 6] < 4).sum().item() / scores.shape[0] * 100
        #print(f"DCC success rate: {dcc_success:.2f}%")
        #print(f"DCA success rate: {dca_success:.2f}%")
    return



test(config_data, config_model, config_runtime, output_file=f"preds_{model2_num}.h5")
