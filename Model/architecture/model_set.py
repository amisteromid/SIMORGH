# Copyright (c) 2026 Omid Mokhtari

import torch
import torch_geometric
from e3nn import o3
from e3nn.nn import Activation, Gate

from architecture.layers import DWTP
from architecture.primitives import EquivariantLayerNormV2 as layernorm
from architecture.primitives import LinearRS, ScaledScatter, irreps2gate


class SetModel(torch.nn.Module):
    def __init__(self, config):
        super(SetModel, self).__init__()

        # Node State Block
        self.irreps_node = o3.Irreps(config["dims"]["_NODE_STATE_IRREPS"])
        self.emb_l0 = next(
            mul for mul, ir in self.irreps_node if ir.l == 0 and ir.p == 1
        )

        # Scale scatter for aggregation
        self.scale_scatter = ScaledScatter(20)

        # Pre pool gate
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node)
        self.lin = LinearRS(
            self.irreps_node, (irreps_scalars + irreps_gates + irreps_gated).simplify()
        )
        self.gate = Gate(
            irreps_scalars,
            [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
            irreps_gates,
            [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated,  # gated tensors
        )

        # attention/agreement scores
        self.dwtp_latent_score = DWTP(
            self.irreps_node,  # input 1: The Node
            self.irreps_node,  # input 2: The Current Latent
            o3.Irreps("1x0e"),  # Output: Scalar weight (Logit for attention)
            bias=True,
        )
        self.mlp_latent_score = LinearRS(
            self.dwtp_latent_score.irreps_out.simplify(),
            o3.Irreps("3x0e"),
            rescale=True,
        )

        # Norm
        self.norm_node = layernorm(self.irreps_node)

        # Invariant operation + MLP
        self.dwtp_decode = DWTP(
            self.irreps_node, self.irreps_node, o3.Irreps("1x0e"), bias=True
        )
        self.decode_mlp = torch.nn.Sequential(
            Activation(self.dwtp_decode.irreps_out.simplify(), acts=[torch.nn.SiLU()]),
            LinearRS(
                self.dwtp_decode.irreps_out.simplify(), o3.Irreps("1x0e"), rescale=True
            ),
        )

    def forward(self, X_n_set):
        # X_n_set shape: [num_res, num_conformations, irreps]

        device = X_n_set.device
        num_res, num_conf, emb_dim = X_n_set.shape
        X_n_flat = X_n_set.view(num_res * num_conf, -1)
        node_index = torch.arange(num_res, device=device).repeat_interleave(
            num_conf
        )  # [0, 0, ..., 1, 1, ...]
        # initialize global token (a boring query as mean of original embeddings)
        token_approx = self.scale_scatter(X_n_flat, node_index, dim=0, dim_size=num_res)

        # mix + gate
        X_n_flat = self.lin(X_n_flat)
        X_n_flat_gated = self.gate(X_n_flat)

        # attention/agreement score
        alpha_logits = self.dwtp_latent_score(X_n_flat_gated, token_approx[node_index])
        alpha_logits = self.mlp_latent_score(alpha_logits)
        alphas = torch_geometric.utils.softmax(alpha_logits, node_index)
        alpha_l0, alpha_l1, alpha_l2 = alphas.unbind(dim=-1)
        alpha_map = {
            0: alpha_l0.unsqueeze(-1),
            1: alpha_l1.unsqueeze(-1),
            2: alpha_l2.unsqueeze(-1),
        }

        # recalculate global token but this time on projected embedding & attention-based
        weighted_slices = []
        idx = 0
        for mul, ir in self.irreps_node:
            dim = mul * ir.dim
            # Grab the slice corresponding to this irrep chunk
            field = X_n_flat_gated[:, idx: idx + dim]
            if ir.l in alpha_map:
                field = field * alpha_map[ir.l]
            weighted_slices.append(field)
            idx += dim
        X_n_flat_weighted = torch.cat(weighted_slices, dim=1)

        token_clean = self.scale_scatter(
            X_n_flat_weighted, node_index, dim=0, dim_size=num_res
        )

        # normalize
        token_clean = self.norm_node(token_clean)

        # Decoding
        scalar_output = self.dwtp_decode(token_clean, token_clean)
        out = self.decode_mlp(scalar_output)

        return out
