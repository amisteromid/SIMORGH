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
            dwtp_output = self.dwtp_decode(X_n, X_n)
            X_n = self.decode_mlp(dwtp_output)

        return X_n
