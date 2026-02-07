# Copyright (c) 2026 Omid Mokhtari

import torch
import numpy as np
import torch.nn as nn
import math


class PolynomialEnvelope(nn.Module):
    """
    Polynomial Envelope function u(x) ensuring C2 continuity at the cutoff.
    p : The exponent base. Typically 5 or 6 in GemNet/DimeNet.
    """
    
    def __init__(self, p: int = 5):
        super().__init__()
        if p <= 0:
            raise ValueError(f"Exponent p must be positive, got {p}")
            
        self.p = p
        
        self.a = -(p + 1) * (p + 2) / 2.0
        self.b = p * (p + 2)
        self.c = -p * (p + 1) / 2.0

    def forward(self, d_scaled: torch.Tensor):
        env_val = (
            1.0
            + self.a * d_scaled.pow(self.p)
            + self.b * d_scaled.pow(self.p + 1)
            + self.c * d_scaled.pow(self.p + 2)
        )

        return torch.where(d_scaled < 1.0, env_val, torch.zeros_like(d_scaled))


class BesselBasisLayer(nn.Module):
    """
    Based on the DimeNet/GemNet formulation: 
    RBF_n(d) = sqrt(2/c) * sin(n * pi * d / c) / d
    
    num_radial : The number of radial basis functions (frequencies).
    cutoff : The cutoff distance in Angstroms.
    envelope_exponent : Exponent for the polynomial envelope function. Default is 5.
    trainable_frequencies : If True, frequencies become learnable parameters. If False (default per paper), they remain fixed at canonical positions n*pi.
    """
    def __init__(
        self, 
        num_radial: int, 
        cutoff: float, 
        envelope_exponent: int = 5,
        trainable_frequencies: bool = False
    ):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = 1.0 / cutoff
        
        self.envelope = PolynomialEnvelope(envelope_exponent)

        self.norm_const = (2.0 * self.inv_cutoff) ** 0.5
        canonical_frequencies = np.pi * torch.arange(1, num_radial + 1, dtype=torch.float32)
        if trainable_frequencies:
            self.frequencies = nn.Parameter(canonical_frequencies)
        else:
            self.register_buffer("frequencies", canonical_frequencies)

    def forward(self, d: torch.Tensor):
        """
        Returns Radial basis features. Shape: (num_edges, num_radial)
        """
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)

        sin_term = torch.sin(self.frequencies * d_scaled)
        
        rbf = self.norm_const * sin_term / d

        return env * rbf
