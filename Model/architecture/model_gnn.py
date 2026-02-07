# Copyright (c) 2026 Omid Mokhtari

import torch
from e3nn import o3
from e3nn.nn import Activation
from architecture.layers import DWTP
from architecture.blocks import NodeStateBlock, StateUpdateLayer
from architecture.primitives import LinearRS

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        # Node State Block
        self.emb = NodeStateBlock(config = config['dims'], bias=True)

        # Sequential State Update Layers
        self.sul = torch.nn.Sequential(*[
            StateUpdateLayer(config['dims'], layer_size) 
            for layer_size in config['layers']
        ])

        # Decoding Layer - Invariant operation + MLP
        self.dwtp_decode = DWTP(
            o3.Irreps(config['dims']['_NODE_STATE_IRREPS']),
            o3.Irreps(config['dims']['_NODE_STATE_IRREPS']),
            o3.Irreps('1x0e'),
            bias=True
        )
        self.decode_mlp = torch.nn.Sequential(
            Activation(self.dwtp_decode.irreps_out.simplify(), acts=[torch.nn.SiLU()]),
            LinearRS(self.dwtp_decode.irreps_out.simplify(), o3.Irreps('1x0e'), rescale=True),
        )

    def forward(self, features, edge_src, edge_dst, get_mor=False):
        '''
        features = [[res_type, SCOV], [SCOD, R, D]]
        '''
        seq, SCOV = features[0]
        SCOD, R, D = features[1]

        # initial node states and raw messages
        X_n = self.emb([seq, SCOV, SCOD, R, D], edge_src, edge_dst)

        # update states
        X_n, _, _, _ = self.sul((X_n, [SCOD, R, D], edge_src, edge_dst))

        if get_mor==False:
            # Decoding Layer
            dwtp_output = self.dwtp_decode(node_features=X_n, edge_attr=X_n)
            X_n = self.decode_mlp(dwtp_output)

        return X_n


















def check_rotation_equivariance(rot, out_orig, out_rot, irreps_out, tolerance=1e-5):
    if isinstance(irreps_out, str):
        irreps_out = o3.Irreps(irreps_out)

    invariants_orig = []
    invariants_rot = []
    
    start_idx = 0
    for mul, ir in irreps_out:
        # Slice the tensor to get the component for the current irrep
        end_idx = start_idx + mul * ir.dim
        component_orig = out_orig[..., start_idx:end_idx].view(*out_orig.shape[:-1], mul, ir.dim)
        component_rot = out_rot[..., start_idx:end_idx].view(*out_rot.shape[:-1], mul, ir.dim)
        
        if ir.l == 0:
            # Condition 1: For scalars, the invariant is the value itself.
            invariants_orig.append(component_orig.squeeze(-1))
            invariants_rot.append(component_rot.squeeze(-1))
        else:
            # Condition 2: For higher irreps, the invariant is the norm.
            norm_orig = torch.norm(component_orig, dim=-1)
            norm_rot = torch.norm(component_rot, dim=-1)
            invariants_orig.append(norm_orig)
            invariants_rot.append(norm_rot)
            
        start_idx = end_idx
        
    # Concatenate all invariants and compare
    all_invariants_orig = torch.cat(invariants_orig, dim=-1)
    all_invariants_rot = torch.cat(invariants_rot, dim=-1)
    
    ok = torch.allclose(all_invariants_orig, all_invariants_rot, atol=tolerance, rtol=tolerance)
    return bool(ok)
if __name__ == '__main__':
    import h5py
    import numpy as np
    from architecture.feature_extraction import aa_idx, minmax_normalize, extract_dynamic_features_knn, extract_topology_knn
    # Initialize model
    model = Model(config_model)
    hf = h5py.File('/home/omokhtar/Desktop/ligand_site/E3DynamiT/Data/raw_xyz_label/ca-ca-coords_and_labels.h5', 'r')['5T5G']
    xyz = hf['coordinates']
    seq = hf['sequence'][()].decode('utf-8')
    seq = torch.tensor([aa_idx[i] for i in seq], dtype=torch.long)
    # Features
    rmsf, motionV, motionV_var, motionS, motionS_var = extract_dynamic_features_knn(xyz)
    R, R_var, D, D_var, SCOV, SCOV_var, SCOD, SCOD_var, nn_ids = extract_topology_knn(xyz)
    R = o3.spherical_harmonics('1x1e', torch.tensor(R), normalize=True, normalization='component')
    SCOD = o3.spherical_harmonics('1x1e', torch.tensor(SCOD), normalize=True, normalization='component')
    SCOV = o3.spherical_harmonics('1x1e', torch.tensor(SCOV), normalize=True, normalization='component')
    # motionV = o3.spherical_harmonics('1x1e', torch.tensor(motionV), normalize=True, normalization='component')
    # R = torch.zeros_like(R)
    # SCOD = torch.zeros_like(SCOD)
    # SCOV = torch.zeros_like(SCOV)
    # motionV = torch.zeros_like(motionV)
    rmsf, motionV_var, motionS, motionS_var,R_var = minmax_normalize(rmsf),minmax_normalize(motionV_var),minmax_normalize(motionS),minmax_normalize(motionS_var),minmax_normalize(R_var)
    D_var, SCOV_var, SCOD_var = minmax_normalize(D_var), minmax_normalize(SCOV_var), minmax_normalize(SCOD_var)
    num_nodes, k = nn_ids.shape
    edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
    edge_dst = nn_ids.flatten()
    torch.manual_seed(123)
    z = model.forward([[torch.tensor(seq), torch.tensor(SCOV), torch.tensor(SCOV_var), torch.tensor(rmsf)], [torch.tensor(motionV), torch.tensor(SCOD), torch.tensor(R), torch.tensor(D), torch.tensor(D_var), torch.tensor(R_var), torch.tensor(motionS), torch.tensor(motionS_var), torch.tensor(motionV_var), torch.tensor(SCOD_var)]],torch.tensor(edge_src),torch.tensor(edge_dst))

    # -------- Apply a random rotation matrix -------- #
    theta = torch.rand(1).item() * 2 * math.pi  # random angle in [0, 2π]
    rot = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0],
        [0, 0, 1]
    ], device='cpu', dtype=torch.float32)
    #print("det(rot):", torch.det(rot))  # Should be +1
    #print("rot @ rot.T close to I:", torch.allclose(rot @ rot.T, torch.eye(3, device=rot.device)))

    # Rotate positions
    xyz_rot = torch.tensor(xyz) @ rot.T
    # Features - vectors: R, SCOV, motionV, SCOD
    rmsf, motionV, motionV_var, motionS, motionS_var = extract_dynamic_features_knn(xyz_rot)
    R, R_var, D, D_var, SCOV, SCOV_var, SCOD, SCOD_var, nn_ids = extract_topology_knn(xyz_rot)
    R = o3.spherical_harmonics('1x1e', torch.tensor(R), normalize=True, normalization='component')
    SCOD = o3.spherical_harmonics('1x1e', torch.tensor(SCOD), normalize=True, normalization='component')
    SCOV = o3.spherical_harmonics('1x1e', torch.tensor(SCOV), normalize=True, normalization='component')
    # motionV = o3.spherical_harmonics('1x1e', torch.tensor(motionV), normalize=True, normalization='component')
    # R = torch.zeros_like(R)
    # SCOD = torch.zeros_like(SCOD)
    # SCOV = torch.zeros_like(SCOV)
    # motionV = torch.zeros_like(motionV)
    rmsf, motionV_var, motionS, motionS_var,R_var = minmax_normalize(rmsf),minmax_normalize(motionV_var),minmax_normalize(motionS),minmax_normalize(motionS_var),minmax_normalize(R_var)
    D_var, SCOV_var, SCOD_var = minmax_normalize(D_var), minmax_normalize(SCOV_var), minmax_normalize(SCOD_var)
    num_nodes, k = nn_ids.shape
    edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
    edge_dst = nn_ids.flatten()
    torch.manual_seed(123)
    z_rot = model.forward([[torch.tensor(seq), torch.tensor(SCOV), torch.tensor(SCOV_var), torch.tensor(rmsf)], [torch.tensor(motionV), torch.tensor(SCOD), torch.tensor(R), torch.tensor(D), torch.tensor(D_var), torch.tensor(R_var), torch.tensor(motionS), torch.tensor(motionS_var), torch.tensor(motionV_var), torch.tensor(SCOD_var)]],torch.tensor(edge_src),torch.tensor(edge_dst))
    #print (torch.sigmoid(z))
    #print (torch.sigmoid(z_rot))
    print (torch.allclose(z,z_rot, atol=1e-3))
