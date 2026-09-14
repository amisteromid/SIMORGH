# Copyright (c) 2026 Omid Mokhtari

from functools import lru_cache

import torch
from e3nn import o3
from e3nn.nn import Gate

from architecture.primitives import (FullyConnectedTensorProductRescale,
                                     LinearRS, ScaleFactor,
                                     TensorProductRescale, Vec2AttnHeads,
                                     irreps2gate, sort_irreps_even_first)


class irreps2scalarsLinear(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        irreps_in2 = irreps_in
        super().__init__(
            irreps_in,
            irreps_in2,
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=True,
            shared_weights=True,
            normalization=None,
        )

    def forward(self, x):
        out = self.forward_tp_rescale_bias(x, x)
        return out


class DWTP(TensorProductRescale):
    """
    Inspired from Depth-wise Tensor Product in equiformer.
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        internal_weights=True,
        bias=True,
        rescale=True,
    ):

        irreps_output, instructions = self._generate_instructions(
            irreps_node_input, irreps_edge_attr, irreps_node_output
        )

        super().__init__(
            irreps_node_input,
            irreps_edge_attr,
            irreps_output,
            instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias,
            rescale=rescale,
        )

    def forward(self, node_features, edge_attr, weight=None):
        return super().forward(node_features, edge_attr, weight)

    @staticmethod
    def _generate_instructions(irreps_node_input, irreps_edge_attr, irreps_node_output):
        irreps_output = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps_node_input):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_node_output:
                        k = len(irreps_output)
                        irreps_output.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_output = o3.Irreps(irreps_output)
        irreps_output, p, *_ = sort_irreps_even_first(irreps_output)
        # Reindex instructions based on sorting
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        return irreps_output, instructions


class irreps2headsLinear(torch.nn.Module):
    def __init__(self, irreps_in, config):
        super().__init__()
        self.irreps_head = o3.Irreps(config["_IRREPS_HEAD"])
        self.num_heads = config["_NUM_HEADS"]
        self.attn_X_heads = (self.irreps_head * self.num_heads).sort()[0].simplify()
        self.query = LinearRS(irreps_in, self.attn_X_heads, bias=False)
        self.vec2heads = Vec2AttnHeads(self.irreps_head, self.num_heads)
        self.scale_factor = ScaleFactor(self.irreps_head)

    def forward(self, attn_component_input):
        attn_component = self.query(attn_component_input)
        attn_component = self.vec2heads(attn_component)
        attn_component = self.scale_factor(attn_component)

        return attn_component


class NodeMessageMixer(torch.nn.Module):
    def __init__(self, node_state_irreps, raw_message_irreps):
        super().__init__()
        self.out_irreps = node_state_irreps
        self.node_scalar_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 0 and ir.p == 1]
        )
        self.node_vectorial_state_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.out_irreps if ir.l == 1 and ir.p == 1]
        )
        self.SRM_irreps, self.VRM_irreps = raw_message_irreps

        self.scalar_value_model = DWTP(
            self.SRM_irreps,
            self.node_scalar_state_irreps,
            self.node_scalar_state_irreps,
            bias=False,
        )
        self.vectorial_value_model = DWTP(
            self.VRM_irreps,
            self.node_vectorial_state_irreps,
            self.node_vectorial_state_irreps,
            bias=False,
        )

        self.message_irreps = (
            self.scalar_value_model.irreps_out.simplify() + self.vectorial_value_model.irreps_out.simplify()
        )
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.message_irreps)
        irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
        irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.message_irreps, irreps_lin_output)
        self.gate = Gate(
            irreps_scalars,
            [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
            irreps_gates,
            [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated,  # gated tensors
        )
        self.proj = LinearRS(self.message_irreps, self.out_irreps)

    def forward(self, node_state, raw_message, edge_src, edge_dst):
        node_states_dic = split_irreps_tensor(node_state[edge_dst], self.out_irreps)
        SV = self.scalar_value_model(raw_message[0], node_states_dic["01"])
        VV = self.vectorial_value_model(raw_message[1], node_states_dic["11"])
        V = torch.concat([SV, VV], dim=-1)
        # Non-linearity
        V = self.lin(V)
        V = self.gate(V)
        V = self.proj(V)

        return V


@lru_cache(maxsize=None)
def _cached_irreps_slices(irreps_str):
    irreps = o3.Irreps(irreps_str)
    return tuple(
        (f"{ir.l}{ir.p}", slice_indices.start, slice_indices.stop)
        for (_, ir), slice_indices in zip(irreps, irreps.slices())
    )


def split_irreps_tensor(tensor, irreps_str):
    """
    Split a tensor based on irreps using the built-in slices() method.
    """
    result = {}
    for key, start, stop in _cached_irreps_slices(str(irreps_str)):
        field = tensor.narrow(1, start, stop - start)
        if key in result:
            # If we already have this irrep type, concatenate
            result[key] = torch.cat([result[key], field], dim=1)
        else:
            result[key] = field

    return result
