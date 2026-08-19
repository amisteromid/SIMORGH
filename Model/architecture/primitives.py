"""
This file contains functions and modules copied and adapted from the Equiformer repository:
https://github.com/atomicarchitects/equiformer

Original license: MIT License
Copyright (c) 2023 Yi-Lun Liao
"""

import collections
import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.math import perm
from e3nn.nn import Activation, Gate
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch_scatter import scatter


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num**0.5)
        return out

    def extra_repr(self):
        return "avg_aggregate_num={}".format(self.avg_aggregate_num)


def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = "0e"
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=instructions,
            normalization=normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            path_normalization="none",
        )

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:

        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):

        out = self.tp(x, y, weight)

        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] = out[:, slice] + bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        instructions = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )


class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(
            irreps_in,
            o3.Irreps("1x0e"),
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=True,
            shared_weights=True,
            normalization=None,
        )

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out


def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (ell, -p)) for ell, p, _, mul in out])
    return Ret(irreps, p, inv)


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out


class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps("{}x0e".format(self.num_irreps))
        )

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out


@compile_mode("script")
class FeedForwardNetwork(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_mlp_mid=None,
        proj_drop=0.1,
    ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_mlp_mid,
            bias=True,
            rescale=True,
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            bias=True,
            rescale=True,
        )

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_output, drop_prob=proj_drop
            )

    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    """

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class ScaleFactor(torch.nn.Module):
    def __init__(self, irreps, normalization="component"):
        super().__init__()
        self.irreps = irreps
        self.channel_norm_factor = 1 / (self.irreps.num_irreps**0.5)

    def forward(self, x):
        out = []
        idx = 0
        for mul, ir in self.irreps:
            field = x.narrow(-1, idx, mul * ir.dim)
            idx += mul * ir.dim
            ir_norm_factor = 1 / (ir.dim**0.5)
            field = field * (self.channel_norm_factor * ir_norm_factor)
            out.append(field)
        out = torch.cat(out, dim=-1)
        return out


class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            # modules.append(nn.ReLU())
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.offset, -bound, bound)

    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
        return f_out


class EquivariantLayerNormV2(nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output
