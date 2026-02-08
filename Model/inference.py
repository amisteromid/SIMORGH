from Bio.Data.PDBData import protein_letters_3to1_extended
from e3nn import o3
import torch
import numpy as np
from glob import glob
import os
import warnings
from sys import argv
from tqdm import tqdm
from architecture.model_gnn import Model
from architecture.model_set import SetModel
from architecture.config import config_model





model1_num = argv[1]
model2_num = argv[2]
device = 'cuda'



k = 32
aa_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'U': 18, 'V': 19, 'W': 20, 'X': 21, 'Y': 22, 'Z': 23}


def _get_ca_cb(atoms):
    """Get CA and CB coords, calculate pseudo-CB for glycine"""
    ca = atoms.get('CA', np.zeros(3))
    if 'CB' in atoms:
        cb = atoms['CB']
    elif 'N' in atoms and 'C' in atoms:
        v = -(atoms['N'] - ca + atoms['C'] - ca)
        cb = ca + v / np.linalg.norm(v) * 1.52
    else:
        cb = np.zeros(3)
    return np.array([ca, cb])
    
def get_xyz(pdb_file):
    """Extract sequence and CA/CB coordinates from multi-model PDB file"""
    all_models = []
    sequence = []
    current_model = []
    
    with open(pdb_file, 'r') as f:
        current_residue = None
        residue_atoms = {}
        current_resname = None
        
        for line in f:
            if line.startswith('MODEL'):
                current_model = []
                sequence = []
                current_residue = None
                residue_atoms = {}
                continue
                
            if line.startswith('ENDMDL'):
                if residue_atoms:
                    current_model.append(_get_ca_cb(residue_atoms))
                    sequence.append(protein_letters_3to1_extended.get(current_resname, 'X'))
                    residue_atoms = {}
                if current_model:
                    all_models.append(np.array(current_model))
                current_residue = None
                continue
            
            if not line.startswith('ATOM'):
                continue
            
            atom_name = line[12:16].strip()
            res_num = int(line[22:26].strip())
            chain = line[21]
            res_key = (res_num, chain)
            resname = line[17:20].strip()
            
            if res_key != current_residue:
                if current_residue and residue_atoms:
                    current_model.append(_get_ca_cb(residue_atoms))
                    sequence.append(protein_letters_3to1_extended.get(current_resname, 'X'))
                current_residue = res_key
                residue_atoms = {}
                current_resname = resname
            
            if atom_name in ('CA', 'CB', 'N', 'C'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_atoms[atom_name] = np.array([x, y, z])
    
    # Handle single model PDB (no MODEL/ENDMDL)
    if len(all_models) < 2:
        warnings.warn(f"{pdb_file} contains {len(all_models)} model(s). Expected >= 2 models.")
        return None, None
    
    coords = np.array(all_models, dtype=np.float32)  # [num_models, num_residues, 2, 3]
    return ''.join(sequence), coords
    
def get_k_nearest_neighbors(xyz_ca, k):
    """Get k nearest neighbors indices for each residue"""
    n_residues = xyz_ca.shape[0]
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
    D = np.linalg.norm(R, axis=-1, keepdims=True)
    # Side chain orientation (CA to CB)
    SCOV = xyz_cb - xyz_ca
    # Side chain orientation difference
    SCOD = SCOV[:, np.newaxis, :] - SCOV[neighbor_indices, :]
    return R, D, SCOV, SCOD, neighbor_indices

from Bio.PDB import PDBParser, PDBIO
def write_pdb_with_bfactor(pdb_file, predictions, output_file=None):
    if output_file is None:
        output_file = pdb_file
    # Convert to numpy if torch tensor
    if hasattr(predictions, 'cpu'):
        predictions = predictions.detach().numpy()
    # Load PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    # Update B-factors for all models
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain):
                for atom in residue:
                    atom.set_bfactor(round(float(predictions[i]), 2))
    # Save
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)


if __name__ == '__main__':
    device = torch.device(device)
    # Setup models
    model1 = Model(config_model).to(device)
    ckpt1 = torch.load(f"model_{model1_num}.pt", map_location=device, weights_only=True)
    model1.load_state_dict(ckpt1["model_state_dict"] if "model_state_dict" in ckpt1 else ckpt1)
    model1.eval()
    model2 = SetModel(config_model).to(device)
    ckpt2 = torch.load(f"model_{model2_num}.pt", map_location=device, weights_only=True)
    model2.load_state_dict(ckpt2["model_state_dict"] if "model_state_dict" in ckpt2 else ckpt2)
    model2.eval()
    
    
    for pdb_file in tqdm(glob(os.path.join("/home/omokhtar/Desktop/SIMORGH/IDR", "*.pdb"))):
        # Setup data
        seq, xyz = get_xyz(pdb_file)
        if seq==None: continue
        seq = torch.tensor([aa_idx[r] for r in seq], dtype=torch.long)
        # GNN
        emb_list = []
        for frame_idx in tqdm(range(len(xyz)), leave=False):
            # Extract features
            R, D, SCOV, SCOD, nn_ids = extract_topology_knn(xyz, frame_idx=frame_idx, k=k)
            # Spherical harmonics
            R = o3.spherical_harmonics('1x1e+1x2e', torch.tensor(R), normalize=True, normalization='component')
            SCOD = o3.spherical_harmonics('1x1e+1x2e', torch.tensor(SCOD), normalize=True, normalization='component')
            SCOV = o3.spherical_harmonics('1x1e+1x2e', torch.tensor(SCOV), normalize=True, normalization='component')
            # Edge
            num_nodes, k = nn_ids.shape
            edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
            edge_dst = torch.tensor(nn_ids.flatten())
            with torch.no_grad():
                emb = model1([[seq.to(device), SCOV.to(device)], [SCOD.to(device), R.to(device), torch.tensor(D).to(device)]],edge_src.to(device),edge_dst.to(device),get_mor=True)
            emb_list.append(emb)
        emb_list = torch.stack(emb_list, dim=1)
        # evalluate with setmodeli
        z = model2.forward(emb_list)
        write_pdb_with_bfactor(pdb_file, torch.sigmoid(z).cpu(), pdb_file.replace('.pdb', '_predicted.pdb'))
        
        
        
        
        
