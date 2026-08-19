# Copyright (c) 2026 Omid Mokhtari

import os
import warnings
from glob import glob
from sys import argv

import numpy as np
import torch
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import PDBIO, PDBParser
from e3nn import o3
from tqdm import tqdm

from architecture.config import config_model
from architecture.model_gnn import Model
from architecture.model_set import SetModel

model1_num = argv[1]
model2_num = argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # Pairwise distances
    diff = xyz_ca[:, np.newaxis, :] - xyz_ca[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    # Exclude self
    distances[np.arange(n_residues), np.arange(n_residues)] = np.inf
    # Get k nearest
    return np.argpartition(distances, k, axis=1)[:, :k]


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


def write_pdb_with_bfactor(pdb_file, predictions, output_file=None):
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

    # ── write back ────────────────────────────────────────────────────────────
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                res_key = (residue.id[1], chain.id)  # (seq_num, chain_id)
                if res_key not in key_to_pred:
                    continue  # insertion code / HETATM etc.
                pred_val = float(key_to_pred[res_key])
                for atom in residue:
                    atom.set_bfactor(round(pred_val, 2))

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)


if __name__ == "__main__":
    device = torch.device(device)
    # Setup models
    model1 = Model(config_model).to(device)
    ckpt1 = torch.load(f"model_{model1_num}.pt", map_location=device, weights_only=True)
    model1.load_state_dict(
        ckpt1["model_state_dict"] if "model_state_dict" in ckpt1 else ckpt1
    )
    model1.eval()
    model2 = SetModel(config_model).to(device)
    ckpt2 = torch.load(f"model_{model2_num}.pt", map_location=device, weights_only=True)
    model2.load_state_dict(
        ckpt2["model_state_dict"] if "model_state_dict" in ckpt2 else ckpt2
    )
    model2.eval()

    sel = np.genfromtxt(
        "/home/omokhtari/SIMORGH/Data/setup3/plinder_test/apo_structures_for_test.txt",
        dtype=np.dtype("U"),
    )[::10]
    print(len(sel))
    # for pdb_file in tqdm(glob(os.path.join("/home/omokhtari/SIMORGH/Data/setup2/native_benchmarks", "*.pdb"))):
    for pdb_file in tqdm(
        glob(
            os.path.join(
                "/srv/storage/delta@storage4.nancy.grid5000.fr/omokhtari/plinder/apo_structures_bbflow_realigned",
                "*.pdb",
            )
        )
    ):
        # for pdb_file in tqdm(glob(os.path.join("/home/omokhtari/SIMORGH/Model/cryptobench_MD","*.pdb"))):
        # for pdb_file in tqdm(glob(os.path.join("/srv/storage/delta@storage4.nancy.grid5000.fr/omokhtari/bbflow_data/all_structures/", "*.pdb"))):
        # print (pdb_file)
        if os.path.basename(pdb_file)[:-4] not in sel:
            continue
        # if "5OJ0_A" in pdb_file: continue
        name = os.path.basename(pdb_file).replace(".pdb", "_predicted.pdb")
        out_file = os.path.join("benchmarks_plinder_SIMORGH3", name)
        if os.path.exists(out_file):
            continue
        # Setup data
        seq, xyz = get_xyz(pdb_file)
        if seq is None:
            continue
        seq = torch.tensor([aa_idx[r] for r in seq], dtype=torch.long)
        # GNN
        emb_list = []
        for frame_idx in tqdm(range(len(xyz)), leave=False):
            # Extract features
            R, D, SCOV, SCOD, nn_ids = extract_topology_knn(
                xyz, frame_idx=frame_idx, k=k
            )
            # Spherical harmonics
            R = o3.spherical_harmonics(
                "1x1e+1x2e", torch.tensor(R), normalize=True, normalization="component"
            )
            SCOD = o3.spherical_harmonics(
                "1x1e+1x2e",
                torch.tensor(SCOD),
                normalize=True,
                normalization="component",
            )
            SCOV = o3.spherical_harmonics(
                "1x1e+1x2e",
                torch.tensor(SCOV),
                normalize=True,
                normalization="component",
            )
            # Edge
            num_nodes, k = nn_ids.shape
            edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
            edge_dst = torch.tensor(nn_ids.flatten())
            with torch.no_grad():
                emb = model1(
                    [
                        [seq.to(device), SCOV.to(device)],
                        [SCOD.to(device), R.to(device), torch.tensor(D).to(device)],
                    ],
                    edge_src.to(device),
                    edge_dst.to(device),
                    get_mor=True,
                )
                # z = model1([[seq.to(device), SCOV.to(device)], [SCOD.to(device), R.to(device), torch.tensor(D).to(device)]],edge_src.to(device),edge_dst.to(device),get_mor=False)
            emb_list.append(emb)
        emb_list = torch.stack(emb_list, dim=1)
        # evalluate with setmodeli
        z = model2.forward(emb_list)
        # print (os.path.basename(pdb_file), len(z), torch.sigmoid(z).flatten())
        write_pdb_with_bfactor(pdb_file, torch.sigmoid(z).cpu(), out_file)
