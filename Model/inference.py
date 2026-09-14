# Copyright (c) 2026 Omid Mokhtari

import argparse
import os
import warnings

import numpy as np
import torch
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import PDBIO, PDBParser
from e3nn import o3
from hdbscan import HDBSCAN
from scipy.spatial import cKDTree
from tqdm import tqdm
import py3Dmol

from architecture.config import config_model
from architecture.model_gnn import Model
from architecture.model_set import SetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device} is being used...")

k = 32
aa_idx = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "U": 18,
    "V": 19,
    "W": 20,
    "X": 21,
    "Y": 22,
    "Z": 23,
}


def _get_ca_cb(atoms):
    """Get CA and CB coords, calculate pseudo-CB for glycine"""
    ca = atoms.get("CA", np.zeros(3))
    if "CB" in atoms:
        cb = atoms["CB"]
    elif "N" in atoms and "C" in atoms:
        v = -(atoms["N"] - ca + atoms["C"] - ca)
        cb = ca + v / np.linalg.norm(v) * 1.52
    else:
        cb = np.zeros(3)
    return np.array([ca, cb])


def get_xyz(pdb_file):
    """Extract sequence and CA/CB coordinates from multi-model PDB file"""
    all_models = []
    sequence = []
    current_model = []

    with open(pdb_file, "r") as f:
        current_residue = None
        residue_atoms = {}
        current_resname = None

        for line in f:
            if line.startswith("MODEL"):
                current_model = []
                sequence = []
                current_residue = None
                residue_atoms = {}
                continue

            if line.startswith("ENDMDL"):
                if residue_atoms:
                    current_model.append(_get_ca_cb(residue_atoms))
                    sequence.append(
                        protein_letters_3to1_extended.get(current_resname, "X")
                    )
                    residue_atoms = {}
                if current_model:
                    all_models.append(np.array(current_model))
                current_residue = None
                continue

            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            res_num = int(line[22:26].strip())
            chain = line[21]
            res_key = (res_num, chain)
            resname = line[17:20].strip()

            if res_key != current_residue:
                if current_residue and residue_atoms:
                    current_model.append(_get_ca_cb(residue_atoms))
                    sequence.append(
                        protein_letters_3to1_extended.get(current_resname, "X")
                    )
                current_residue = res_key
                residue_atoms = {}
                current_resname = resname

            if atom_name in ("CA", "CB", "N", "C"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_atoms[atom_name] = np.array([x, y, z])

    # Flush last residue and model for single-model PDBs (no ENDMDL marker)
    if current_model or residue_atoms:
        if residue_atoms:
            current_model.append(_get_ca_cb(residue_atoms))
            sequence.append(protein_letters_3to1_extended.get(current_resname, "X"))
        if current_model:
            all_models.append(np.array(current_model))

    if len(all_models) < 1:
        warnings.warn(f"{pdb_file} contains 0 models.")
        return None, None

    # for i, model in enumerate(all_models):
    #     try:
    #         print (i, np.array(model).shape)
    #     except Exception as e:
    #         print(i, e)
    coords = np.array(all_models, dtype=np.float32)  # [num_models, num_residues, 2, 3]
    return "".join(sequence), coords


def get_k_nearest_neighbors(xyz_ca, k):
    """Get k nearest neighbors indices for each residue"""
    n_residues = xyz_ca.shape[0]
    k = min(k, n_residues - 1)
    if k <= 0:
        return np.empty((n_residues, 0), dtype=np.int64)

    # KD-tree query avoids materializing the full NxN distance matrix, which is
    # noticeably faster and lighter for CPU inference on larger proteins.
    _, indices = cKDTree(xyz_ca).query(xyz_ca, k=k + 1)
    neighbors = np.empty((n_residues, k), dtype=np.int64)
    for row_idx, row in enumerate(indices):
        neighbors[row_idx] = row[row != row_idx][:k]
    return neighbors


def extract_topology_knn(xyz, frame_idx, k):
    """
    Extract topology features for k-nearest neighbors.
    xyz: [n_frames, n_residues, 2, 3]
    Returns: R, D, SCOV, SCOD, neighbor_indices
    """
    xyz_ca = xyz[frame_idx, :, 0, :]  # [n_residues, 3]
    xyz_cb = xyz[frame_idx, :, 1, :]  # [n_residues, 3]
    # Get neighbors
    neighbor_indices = get_k_nearest_neighbors(xyz_ca, k)
    # CA-CA displacement vectors [n_residues, k, 3]
    R = xyz_ca[:, np.newaxis, :] - xyz_ca[neighbor_indices, :]
    # CA-CA distances [n_residues, k, 1]
    D = np.linalg.norm(R, axis=-1, keepdims=True).astype(np.float32)
    # Side chain orientation (CA to CB)
    SCOV = xyz_cb - xyz_ca
    # Side chain orientation difference
    SCOD = SCOV[:, np.newaxis, :] - SCOV[neighbor_indices, :]
    return R, D, SCOV, SCOD, neighbor_indices


def write_pdb_with_bfactor(pdb_file, predictions, output_file=None, cluster_ids=None, cluster_output_file=None):
    if output_file is None:
        output_file = pdb_file

    if hasattr(predictions, "cpu"):
        predictions = predictions.detach().numpy()

    # ── rebuild the same ordered key list that get_xyz produced ──────────────
    ordered_keys = []
    seen = set()
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                ordered_keys = []  # only care about first model
                seen = set()
            if line.startswith("ENDMDL"):
                break  # stop after first model
            if not line.startswith("ATOM"):
                continue
            res_num = int(line[22:26].strip())
            chain = line[21]
            res_key = (res_num, chain)
            if res_key not in seen:
                seen.add(res_key)
                ordered_keys.append(res_key)

    key_to_pred = {key: predictions[i] for i, key in enumerate(ordered_keys)}
    key_to_cluster = None
    if cluster_ids is not None:
        # noise (-1) -> 0, real clusters (0,1,2,...) -> 1,2,3,...
        key_to_cluster = {
            key: (0 if cluster_ids[i] == -1 else int(cluster_ids[i]) + 1)
            for i, key in enumerate(ordered_keys)
        }

    def _write(bfactor_map, out_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" not in residue:
                        continue
                    res_key = (residue.id[1], chain.id)
                    if res_key not in bfactor_map:
                        continue
                    val = float(bfactor_map[res_key])
                    for atom in residue:
                        atom.set_bfactor(round(val, 2))
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_path)

    _write(key_to_pred, output_file)

    if key_to_cluster is not None:
        if cluster_output_file is None:
            root, ext = os.path.splitext(output_file)
            cluster_output_file = f"{root}_clusters{ext}"
        _write(key_to_cluster, cluster_output_file)


def cluster_predictions(xyz, probs, cutoff=0.4, min_cluster_size=25, min_samples=10, alpha=1.5):
    """
    Cluster high-probability residues using HDBSCAN with a probability-weighted
    distance metric.  Returns an array of cluster IDs (int) with the same length
    as the number of residues.  -1 means noise / not assigned.

    xyz   : [n_residues, 2, 3]  CA/CB coords for the first frame
    probs : [n_residues]        sigmoid probabilities (0-1)
    """
    probs = np.asarray(probs).flatten()
    n_res = xyz.shape[0]

    labels = np.full(n_res, -1, dtype=int)

    # ── keep only residues above the cutoff ──────────────────────────────────
    mask = probs > cutoff
    if mask.sum() == 0:
        warnings.warn(f"No residues above cutoff {cutoff}; skipping clustering.")
        return labels

    coords = xyz[mask, 0, :]  # CA coords [n_selected, 3]
    sel_probs = probs[mask]   # [n_selected]

    # ── probability-weighted distance matrix ─────────────────────────────────
    coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    spatial_dist = np.linalg.norm(coord_diff, axis=-1)
    prob_dist = alpha / np.outer(sel_probs, sel_probs)
    dist = (spatial_dist + prob_dist).astype(np.float64, copy=False)

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    sub_labels = clusterer.fit_predict(dist)

    # ── scatter labels back to full residue array ────────────────────────────
    sel_idx = np.where(mask)[0]
    labels[sel_idx] = sub_labels
    return labels


def render_pdb_html(prob_pdb, cluster_pdb=None, output_html=None,
                    colors=None, width=600, height=500, style='sphere'):
    """
    Build a standalone HTML file with py3Dmol view(s):
      - always: predicted probability (B-factor gradient)
      - if cluster_pdb is given: cluster assignment side-by-side
        (discrete colors, cluster 0 = noise = gray)

    style: 'sphere', 'surface', or 'cartoon'
    """
    if colors is None:
        colors = [
            '#FF3333',  # Vivid Red
            '#1E90FF',  # Dodger Blue
            '#FF1493',  # Deep Pink
            '#FFD700',  # Gold
            '#00CED1',  # Dark Turquoise
            '#32CD32',  # Lime Green
            '#FF8C00',  # Dark Orange
            '#9932CC',  # Dark Orchid
        ]

    with open(prob_pdb) as f:
        prob_data = f.read()

    # ── probability view ──────────────────────────────────────────────────
    view1 = py3Dmol.view(width=width, height=height)
    view1.addModel(prob_data, 'pdb')
    if style == 'surface':
        view1.setStyle({}, {'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'linear_#3939b8_#d3d4d4_#e93939', 'min': 0, 'max': 1}}})
        view1.addSurface(py3Dmol.VDW, {'colorscheme': {'prop': 'b', 'gradient': 'linear_#3939b8_#d3d4d4_#e93939', 'min': 0, 'max': 1}})
    else:
        view1.setStyle(
            {},
            {style: {
                'colorscheme': {
                    'prop': 'b',
                    'gradient': 'linear_#3939b8_#d3d4d4_#e93939',
                    'min': 0,
                    'max': 1
                }
            }}
        )
    view1.zoomTo()

    prob_block = f"""
  <div>
    <h3 style="text-align:center;">Predicted Probability</h3>
    {view1._make_html()}
  </div>"""

    # ── cluster view (only if cluster_pdb provided) ─────────────────────────
    cluster_block = ""
    if cluster_pdb is not None:
        with open(cluster_pdb) as f:
            cluster_data = f.read()

        # collect distinct cluster ids present (0 = noise)
        cluster_vals = set()
        for line in cluster_data.splitlines():
            if line.startswith("ATOM"):
                cluster_vals.add(int(float(line[60:66])))
        cluster_vals = sorted(cluster_vals)

        view2 = py3Dmol.view(width=width, height=height)
        view2.addModel(cluster_data, 'pdb')
        if style == 'surface':
            view2.setStyle({'b': 0}, {'cartoon': {'color': '#808080'}})
            view2.addSurface(py3Dmol.VDW, {'color': '#808080'}, {'b': 0})
            for cid in cluster_vals:
                if cid == 0:
                    continue
                color = colors[(cid - 1) % len(colors)]
                view2.addStyle({'b': cid}, {'cartoon': {'color': color}})
                view2.addSurface(py3Dmol.VDW, {'color': color}, {'b': cid})
        else:
            view2.setStyle({'b': 0}, {style: {'color': '#808080'}})  # noise
            for cid in cluster_vals:
                if cid == 0:
                    continue
                color = colors[(cid - 1) % len(colors)]
                view2.addStyle({'b': cid}, {style: {'color': color}})
        view2.zoomTo()

        cluster_block = f"""
  <div>
    <h3 style="text-align:center;">Clusters</h3>
    {view2._make_html()}
  </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SIMORGH Visualization</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div style="display:flex; gap:20px;">{prob_block}{cluster_block}
</div>
</body>
</html>
"""
    with open(output_html, "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIMORGH inference on a PDB file")
    parser.add_argument("--model1", required=True, help="Path to the GNN model checkpoint (.pt)")
    parser.add_argument("--model2", required=True, help="Path to the SetModel checkpoint (.pt)")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output", required=True, help="Output PDB file (with predictions in B-factor)")
    parser.add_argument("--cluster", action="store_true", help="Run HDBSCAN clustering on predictions and write cluster IDs to occupancy")
    parser.add_argument("--cutoff", type=float, default=0.4, help="Probability cutoff for clustering (default: 0.4)")
    parser.add_argument("--min-cluster-size", type=int, default=25, help="HDBSCAN min_cluster_size (default: 25)")
    parser.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples (default: 10)")
    parser.add_argument("--alpha", type=float, default=1.5, help="Probability weight in custom distance metric (default: 1.5)")
    parser.add_argument("--visualize", action="store_true", help="Generate an HTML visualization of predictions (and clusters if --cluster is set)")
    parser.add_argument("--viz-style", choices=["sphere", "surface", "cartoon"], default="sphere", help="Visualization style (default: sphere)")
    parser.add_argument("--viz-output", type=str, default=None, help="Output HTML file path (default: <output_basename>_viz.html)")
    args = parser.parse_args()

    device = torch.device(device)
    # Setup models
    model1 = Model(config_model).to(device)
    ckpt1 = torch.load(args.model1, map_location=device, weights_only=True)
    model1.load_state_dict(
        ckpt1["model_state_dict"] if "model_state_dict" in ckpt1 else ckpt1
    )
    model1.eval()
    model2 = SetModel(config_model).to(device)
    ckpt2 = torch.load(args.model2, map_location=device, weights_only=True)
    model2.load_state_dict(
        ckpt2["model_state_dict"] if "model_state_dict" in ckpt2 else ckpt2
    )
    model2.eval()

    pdb_file = args.input
    out_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    # Setup data
    seq, xyz = get_xyz(pdb_file)
    if seq is None:
        raise ValueError(f"Could not parse any models from {pdb_file}")
    seq = torch.tensor([aa_idx[r] for r in seq], dtype=torch.long, device=device)
    # GNN
    emb_list = []
    with torch.inference_mode():
        for frame_idx in tqdm(range(len(xyz)), leave=False):
            # Extract features
            R, D, SCOV, SCOD, nn_ids = extract_topology_knn(
                xyz, frame_idx=frame_idx, k=k
            )
            R = torch.from_numpy(R).to(device=device, dtype=torch.float32)
            D = torch.from_numpy(D).to(device=device, dtype=torch.float32)
            SCOV = torch.from_numpy(SCOV).to(device=device, dtype=torch.float32)
            SCOD = torch.from_numpy(SCOD).to(device=device, dtype=torch.float32)
            nn_ids = torch.from_numpy(nn_ids).to(device=device, dtype=torch.long)

            # Spherical harmonics
            R = o3.spherical_harmonics(
                "1x1e+1x2e", R, normalize=True, normalization="component"
            )
            SCOD = o3.spherical_harmonics(
                "1x1e+1x2e",
                SCOD,
                normalize=True,
                normalization="component",
            )
            SCOV = o3.spherical_harmonics(
                "1x1e+1x2e",
                SCOV,
                normalize=True,
                normalization="component",
            )
            # Edge
            num_nodes, num_neighbors = nn_ids.shape
            edge_src = (
                torch.arange(num_nodes, device=device)
                .unsqueeze(1)
                .expand(num_nodes, num_neighbors)
                .reshape(-1)
            )
            edge_dst = nn_ids.reshape(-1)

            if len(xyz) == 1:
                z = model1(
                    [
                        [seq, SCOV],
                        [SCOD, R, D],
                    ],
                    edge_src,
                    edge_dst,
                    get_mor=False,
                )
            else:
                emb = model1(
                    [
                        [seq, SCOV],
                        [SCOD, R, D],
                    ],
                    edge_src,
                    edge_dst,
                    get_mor=True,
                )
                emb_list.append(emb)
        if len(xyz) > 1:
            emb_list = torch.stack(emb_list, dim=1)
            # evaluate with setmodel
            z = model2.forward(emb_list)

    probs = torch.sigmoid(z).detach().cpu().numpy().flatten()

    # ── optional HDBSCAN clustering ────────────────────────────────────────────
    cluster_ids = None
    if args.cluster:
        cluster_ids = cluster_predictions(
            xyz[0],               # use first frame's CA/CB coords
            probs,
            cutoff=args.cutoff,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            alpha=args.alpha,
        )
        n_clusters = len(set(cluster_ids) - {-1})
        print(f"Clustering: {n_clusters} binding-site cluster(s) found "
              f"({(cluster_ids == -1).sum()} residue(s) as noise)")

    write_pdb_with_bfactor(pdb_file, probs, out_file, cluster_ids=cluster_ids)
    print(f"Predictions written to {out_file}")

    # ── optional HTML visualization ────────────────────────────────────────
    if args.visualize:
        root, ext = os.path.splitext(out_file)
        viz_output = args.viz_output or f"{root}_viz.html"
        cluster_out_file = f"{root}_clusters{ext}" if args.cluster else None
        render_pdb_html(out_file, cluster_out_file, viz_output, style=args.viz_style)
        print(f"Visualization written to {viz_output}")
