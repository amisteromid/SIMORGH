
import torch
from e3nn import o3

from architecture.layers import DWTP


class SetModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.irreps_node = o3.Irreps(config["dims"]["_NODE_STATE_IRREPS"])
        nhead = config.get("nhead", 4)
        num_sab = config.get("num_sab", 2)

        # ── 1. Equivariant → scalar via DWTP (self-interaction) ───────────────
        self.to_scalar = DWTP(
            self.irreps_node, self.irreps_node, o3.Irreps("1x0e"), bias=True
        )
        # d_model is whatever DWTP outputs (all scalars, so .simplify() gives Nx0e)
        scalar_irreps = self.to_scalar.irreps_out.simplify()
        d_model = scalar_irreps.dim  # total scalar dimension
        ff_dim = config.get("ff_dim", d_model * 4)

        # ── 2. SAB blocks ──────────────────────────────────────────────────────
        sab_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sab = torch.nn.TransformerEncoder(sab_layer, num_layers=num_sab)

        # ── 3. PMA ────────────────────────────────────────────────────────────
        self.pma_seed = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.pma_attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.pma_ff = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, ff_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ff_dim, d_model),
        )

        # ── 4. Output MLP ─────────────────────────────────────────────────────
        self.out_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_model // 2, 1),
        )

    def forward(self, X_n_set):
        # X_n_set: [num_res, num_conf, irreps_dim]
        num_res, num_conf, _ = X_n_set.shape

        # 1. Project to scalars via DWTP self-interaction
        X_flat = X_n_set.view(num_res * num_conf, -1)
        X_scalar = self.to_scalar(X_flat, X_flat)  # [R*C, d_model]
        X_scalar = X_scalar.view(num_res, num_conf, -1)  # [R, C, d_model]

        # 2. SAB: conformations attend to each other (per residue)
        X_enc = self.sab(X_scalar)  # [R, C, d_model]

        # 3. PMA: pool conformation set → one vector per residue
        seed = self.pma_seed.expand(num_res, -1, -1)  # [R, 1, d_model]
        pooled, _ = self.pma_attn(seed, X_enc, X_enc)  # [R, 1, d_model]
        pooled = pooled.squeeze(1)  # [R, d_model]
        pooled = pooled + self.pma_ff(pooled)  # residual

        # 4. Decode
        return self.out_mlp(pooled)  # [R, 1]
