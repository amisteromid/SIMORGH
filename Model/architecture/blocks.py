# Copyright (c) 2026 Omid Mokhtari

import torch
import torch_geometric
from e3nn import o3
from torch_scatter import scatter

from architecture.layers import (DWTP, NodeMessageMixer, irreps2headsLinear,
                                 split_irreps_tensor)
from architecture.primitives import AttnHeads2Vec
from architecture.primitives import EquivariantLayerNormV2 as layernorm
from architecture.primitives import (FeedForwardNetwork,
                                     FullyConnectedTensorProductRescale,
                                     LinearRS, RadialProfile, ScaledScatter,
                                     Vec2AttnHeads)
from architecture.rbf import BesselBasisLayer


class NodeStateBlock(torch.nn.Module):
    def __init__(self, config, bias=True):
        super().__init__()

        self.irreps_node_embedding = o3.Irreps(config["_NODE_STATE_IRREPS"])
        self.max_residue_type = config["_MAX_RESIDUE_TYPE"]
        self.edge_attr_irreps = o3.Irreps(config["_EDGE_ATTR_IRREPS"])
        self.emb_dim = [
            mul for mul, ir in self.irreps_node_embedding if ir.l == 0 and ir.p == 1
        ][0]
        self.cutoff = 12
        # Node embedding
        self.residue_embedding = torch.nn.Embedding(self.max_residue_type, self.emb_dim)
        torch.nn.init.uniform_(
            self.residue_embedding.weight.data,
            -1.0 / (self.emb_dim**0.5),
            1.0 / (self.emb_dim**0.5),
        )

        # Edge degree embedding
        self.rbf = BesselBasisLayer(
            num_radial=config["_NUM_RADIAL"], cutoff=self.cutoff
        )
        self.dw = DWTP(
            self.irreps_node_embedding,
            o3.Irreps("1x1e+1x2e"),
            self.irreps_node_embedding,
            internal_weights=False,
            bias=False,
        )
        self.rad = RadialProfile([config["_NUM_RADIAL"], self.dw.tp.weight_numel])
        for slice, slice_sqrt_k in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), self.irreps_node_embedding)
        self.scale_scatter = ScaledScatter(27)  # average number of neighbors within 5A

        # Fix: Proper irreps string format
        input_irreps = o3.Irreps(config["_NODE_FEATURES_IRREPS"])
        self.residue2node = FullyConnectedTensorProductRescale(
            irreps_in1=input_irreps,
            irreps_in2=input_irreps,
            irreps_out=self.irreps_node_embedding,
            bias=bias,
            rescale=False,
        )

    def forward(self, node_features, edge_src, edge_dst):
        res_type, SCOV, SCOD, R, D = node_features
        # Learnable embedding of residue types + RMSF + SCOV
        res_type_embedding = self.residue_embedding(res_type)
        res_embedding = torch.cat((SCOV, res_type_embedding), dim=1).float()
        # Map to target node embedding irreps
        node_embedding = self.residue2node(res_embedding, res_embedding)

        # edge
        SCOD = SCOD.view(-1, SCOD.shape[-1])  # Flatten
        R = R.view(-1, R.shape[-1])  # Flatten
        D = D.view(-1, D.shape[-1])  # Flatten
        # Filter edges
        distances = D.squeeze(-1)
        edge_mask = (distances <= self.cutoff).to(edge_src.device)
        if edge_mask.sum() == 0:
            return torch.zeros_like(res_type, dtype=torch.float)  # No valid edges
        filtered_edge_src = edge_src[edge_mask]
        filtered_edge_dst = edge_dst[edge_mask]
        # filtered_R = R[edge_mask] # never used
        filtered_SCOD = SCOD[edge_mask]
        filtered_D = D[edge_mask]
        # edge
        weights = self.rbf(filtered_D)
        weights = self.rad(weights)
        edge_features = self.dw(
            node_embedding[filtered_edge_src], filtered_SCOD, weights
        )
        edge_features = self.proj(edge_features)
        scattered_edge_features = self.scale_scatter(
            edge_features, filtered_edge_dst, dim=0, dim_size=node_embedding.shape[0]
        )
        return node_embedding + scattered_edge_features


class RawMessageBlock(torch.nn.Module):
    def __init__(self, config, max_radius):
        super().__init__()
        self.max_radius = max_radius
        # === Parse node irreps and separate based on l
        self.out_irreps = o3.Irreps(config["_NODE_STATE_IRREPS"])
        self.node_scalar_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 0 and ir.p == 1]
        )
        self.node_vectorial_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 1 and ir.p == 1]
        )

        # === Parse edge irreps (list of 2-strings)
        self.edge_attr_irreps = o3.Irreps(config["_EDGE_ATTR_IRREPS"])

        # Tensor product layers for spatial and dynamic features
        self.rbf = BesselBasisLayer(
            num_radial=config["_NUM_RADIAL"], cutoff=self.max_radius
        )
        self.vectorial_tensor_product = DWTP(
            self.node_vectorial_state_irreps,
            self.edge_attr_irreps,
            self.out_irreps,
            internal_weights=False,
            bias=False,
        )
        self.rad = RadialProfile(
            [
                config["_NUM_RADIAL"],
                config["_NUM_RADIAL"],
                self.vectorial_tensor_product.tp.weight_numel,
            ]
        )
        for slice, slice_sqrt_k in self.vectorial_tensor_product.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k

    def forward(self, X_n, edge_attr, edge_src, edge_dst):
        # Extract nearest neighbor states of destination nodes for each edge
        nn_states = X_n[edge_src]
        # Node state separation
        nn_states_dic = split_irreps_tensor(nn_states, self.out_irreps)
        nn_scalar_state, nn_vectorial_state = nn_states_dic["01"], nn_states_dic["11"]
        # Edge attributes
        weights = self.rbf(edge_attr[2])
        weights = self.rad(weights)
        edge_attr = torch.cat(edge_attr[:2], dim=-1)
        # Scalar Raw Message
        SRM = nn_scalar_state.clone()
        # Vectorial Raw Message
        VRM = self.vectorial_tensor_product(nn_vectorial_state, edge_attr, weights)
        return SRM, VRM


class AttnMPBlock(torch.nn.Module):
    def __init__(self, config, raw_message_irreps):
        super().__init__()
        # === Parse node irreps and separate based on l and parity
        self.out_irreps = o3.Irreps(config["_NODE_STATE_IRREPS"])
        self.node_scalar_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 0 and ir.p == 1]
        )
        self.node_vectorial_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 1 and ir.p == 1]
        )
        self.SRM_irreps, self.VRM_irreps = raw_message_irreps
        self.raw_message_irreps = raw_message_irreps[0] + raw_message_irreps[1]

        self.q_calc = irreps2headsLinear(self.out_irreps, config)
        self.k_calc = irreps2headsLinear(self.raw_message_irreps, config)
        self.v_calc = NodeMessageMixer(
            self.out_irreps, (self.SRM_irreps, self.VRM_irreps)
        )
        # print ((self.out_irreps*config['_NUM_HEADS']).sort()[0].simplify())

        # Add head-structuring layer for value
        self.out_irreps_per_head = o3.Irreps(
            [(mul // config["_NUM_HEADS"], ir) for mul, ir in self.out_irreps]
        ).simplify()
        self.vec2heads_v = Vec2AttnHeads(self.out_irreps_per_head, config["_NUM_HEADS"])
        self.heads2vec = AttnHeads2Vec(self.out_irreps_per_head)

    def forward(self, node_state, raw_message, edge_src, edge_dst):
        # Dot product attention
        q = self.q_calc(node_state)  # [N, _NUM_HEADS, irreps_head]
        concatenated_raw_message = torch.cat(raw_message, dim=-1)
        k = self.k_calc(concatenated_raw_message)  # [E, _NUM_HEADS, irreps_head]
        # DP attention
        alpha = torch.einsum("bik, bik -> bi", q[edge_dst], k)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)  # [E, _NUM_HEADS, 1]

        # Value calculation
        v = self.v_calc(
            node_state, raw_message, edge_src, edge_dst
        )  # [E, _NODE_STATE_IRREPS]
        v = self.vec2heads_v(v)  # [E, _NUM_HEADS, _NODE_STATE_IRREPS/NUM_HEADS]
        # print ('alpha :', alpha.shape)
        # print ('value :',v.shape)

        # Message calculation
        M = v * alpha  # [E, _NUM_HEADS, _NODE_STATE_IRREPS/NUM_HEADS]
        M = scatter(
            M, index=edge_dst, dim=0, dim_size=node_state.shape[0]
        )  # [E, _NUM_HEADS, _NODE_STATE_IRREPS/NUM_HEADS]
        M = self.heads2vec(M)  # [E, _NODE_STATE_IRREPS]

        node_state = node_state + M

        return node_state


class StateUpdateLayer(torch.nn.Module):
    def __init__(self, config_dims, cutoff):
        super(StateUpdateLayer, self).__init__()
        self.cutoff = cutoff

        # Raw Message Block
        self.raw_message = RawMessageBlock(config=config_dims, max_radius=cutoff)
        self.SRM_irreps = self.raw_message.node_scalar_state_irreps
        self.VRM_irreps = (
            self.raw_message.vectorial_tensor_product.irreps_out.simplify()
        )
        # print('SRM -->',self.SRM_irreps, '\nVRM -->',self.DVRM_irreps)

        # Normalize components
        self.norm_node1 = layernorm(config_dims["_NODE_STATE_IRREPS"])
        self.norm_srm = layernorm(self.SRM_irreps)
        self.norm_vrm = layernorm(self.VRM_irreps)

        # Transformer Block
        self.raw_message_irreps = (self.SRM_irreps, self.VRM_irreps)
        self.MP = AttnMPBlock(config_dims, self.raw_message_irreps)

        # Normalize Message
        self.norm_node2 = layernorm(config_dims["_NODE_STATE_IRREPS"])

        # Feed Forward Network
        self.ffn = FeedForwardNetwork(
            irreps_node_input=config_dims["_NODE_STATE_IRREPS"],
            irreps_node_attr=o3.Irreps("0e"),
            irreps_node_output=config_dims["_NODE_STATE_IRREPS"],
            irreps_mlp_mid=config_dims["_NODE_STATE_IRREPS"],
            proj_drop=0.1,
        )

        # Normalize Final
        self.norm_node3 = layernorm(config_dims["_NODE_STATE_IRREPS"])

    def forward(self, input):
        X_n, features, edge_src, edge_dst = input

        # Filter edges and features based on distance cutoff
        distances = features[2].squeeze(-1)
        edge_mask = (distances <= self.cutoff).flatten().to(edge_src.device)
        if edge_mask.sum() == 0:
            return (X_n, features, edge_src, edge_dst)
        filtered_edge_src = edge_src[edge_mask]
        filtered_edge_dst = edge_dst[edge_mask]
        filtered_features_vectorial = []
        for feat in features:
            feat = feat.contiguous()
            filtered_features_vectorial.append(feat.view(-1, feat.shape[-1])[edge_mask])

        # Raw Message Block
        X_e_s, X_e_v = self.raw_message(
            X_n, filtered_features_vectorial[:], filtered_edge_src, filtered_edge_dst
        )

        # Normalize 1
        X_n = self.norm_node1(X_n)
        X_e_s = self.norm_srm(X_e_s)
        X_e_v = self.norm_vrm(X_e_v)

        # Message passing
        X_n = self.MP(X_n, (X_e_s, X_e_v), filtered_edge_src, filtered_edge_dst)

        # Normalize 2
        X_n = self.norm_node2(X_n)

        # Feed Forward Network
        node_dummie = torch.ones_like(X_n.narrow(1, 0, 1))
        X_n = self.ffn(X_n, node_dummie)

        # Normalize 3
        X_n = self.norm_node3(X_n)

        return (X_n, features, edge_src, edge_dst)
