from typing import Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear

from .utils import clip_outliers, flash_context, normalize_data


class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        enc_cell_dim: int,
        n_out: int,
        regression_bin_count: int,
        regression_bin_min: float,
        regression_bin_max: float,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        base_len: int,
        max_len: int,
        y_encoder_dim: int,
        num_col_attn_layers: int = 2,
        n_thinking_rows: int = 0,
        use_flash: bool = True,
        clip_sigma: float = 8.
    ):
        super().__init__()
        assert regression_bin_count >= 3, "regression_bin_count must be at least 3"
        assert regression_bin_min < regression_bin_max, "regression_bin_min must be smaller than regression_bin_max"
        self.n_out = n_out  # number of classification outputs
        self.regression_bin_count = regression_bin_count
        self.regression_bin_min = regression_bin_min
        self.regression_bin_max = regression_bin_max
        self.ninp = ninp  # embedding dimension
        self.nlayers = nlayers  # store for stochastic depth calculation
        self.n_thinking_rows = n_thinking_rows
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                    base_len=base_len,
                    max_len=max_len,
                    y_encoder_dim=y_encoder_dim,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.enc_norm = nn.LayerNorm(ninp, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_y_encoders = nn.ModuleList([nn.Embedding(n_out, y_encoder_dim) for _ in range(nlayers)])
        self.reg_y_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, y_encoder_dim),
                    nn.GELU(),
                    nn.Linear(y_encoder_dim, y_encoder_dim),
                    nn.LayerNorm(y_encoder_dim, elementwise_affine=False),
                )
               for _ in range(nlayers)
            ]
        )
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + regression_bin_count))
        if n_thinking_rows > 0:
            self.thinking_embed = nn.Parameter(torch.empty(n_thinking_rows, ninp))
            nn.init.normal_(self.thinking_embed, std=0.02)
        self.use_flash = use_flash
        self.clip_sigma = clip_sigma

    @flash_context
    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        is_cls: bool
    ) -> torch.Tensor:
        """Forward pass of the TabDPTModel.

        Args:
            x_src: Input features with dimensions (B, T, F). B is the ordinary batch dimension while T is the
                transformer sequence dimension and F is the feature dimension. When ordinary batching is used, this is
                of shape `(n_batch, n_ctx + 1, n_features)`, and when TabPFN-style batching is used, it is of shape
                `(1, n_ctx + n_eval, n_features)`.
            y_src: Target values with dimensions (B, T). The shape is `(n_batch, n_ctx)` when ordinary batching is used and
                `(1, n_ctx)` when TabPFN-style batching is used.
            is_cls: True for classification, False for regression

        Returns:
            torch.Tensor: Predicted values with dimensions (T, B, O). The T and B dimensions match `x_src` but with the
                T dimension reduced by `n_ctx`. The O dimension is of shape `n_out + regression_bin_count`: the number
                of classification plus regression outputs.
        """
        x_src = x_src.transpose(0, 1)
        y_src = y_src.transpose(0, 1)
        eval_pos = y_src.shape[0]
        n_think = self.n_thinking_rows

        # preproces features by normalizing and clipping outliers
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        # Replace NaN and Inf with 0 (inf can occur from division by zero in normalize_data)
        x_src = torch.nan_to_num(x_src, nan=0.0, posinf=0.0, neginf=0.0)

        x_src = self.encoder(x_src)
        src = self.enc_norm(x_src)
        if n_think > 0:
            B = src.shape[1]
            src = torch.cat([self.thinking_embed.unsqueeze(1).expand(n_think, B, -1), src], dim=0)

        for l, layer in enumerate(self.transformer_encoder):
            if is_cls:
                y_cls = y_src.long().clamp(min=0, max=self.n_out - 1)
                y_emb = self.cls_y_encoders[l](y_cls)
            else:
                y_emb = self.reg_y_encoders[l](y_src.unsqueeze(-1))
            if n_think > 0:
                B = y_emb.shape[1]
                y_emb = torch.cat([y_emb.new_zeros(n_think, B, y_emb.shape[-1]), y_emb], dim=0)
            # Layer returns residual only
            residual = layer(src, y_emb, eval_pos + n_think)
            src = src + residual

        # final head
        pred = self.head(src[eval_pos + n_think :].float())
        return pred

    @classmethod
    def load(cls, model_state, config, use_flash, clip_sigma: float = 8.):
        assert config.model.max_num_classes > 2
        model = TabDPTModel(
            dropout=config.training.dropout,
            enc_cell_dim=config.model.enc_cell_dim,
            n_out=config.model.max_num_classes,
            regression_bin_count=config.model.regression_bin_count,
            regression_bin_min=config.model.regression_bin_min,
            regression_bin_max=config.model.regression_bin_max,
            nhead=config.model.nhead,
            nhid=config.model.ff_dim,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            base_len=config.model.min_eval_context,
            max_len=config.model.max_eval_context,
            y_encoder_dim=config.model.y_encoder_dim,
            num_col_attn_layers=config.model.num_col_attn_layers,
            n_thinking_rows=config.model.n_thinking_rows,
            use_flash=use_flash,
            clip_sigma=clip_sigma
        )

        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(config.env.device)
        model.eval()
        return model


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for computation to avoid bf16 underflow/NaNs
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, bias: bool = False):
        super().__init__()
        self.up = Linear(d_model, 2 * ff_dim, bias=bias)
        self.down = Linear(ff_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(u) * v)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        base_len: int,
        max_len: int,
        y_encoder_dim: int,
    ) -> None:
        """Custom transformer encoder layer

        Args:
            embed_dim: Dimension of the embedding.
            num_heads: Number of attention heads.
            ff_dim: Dimension of the feed-forward network.
            base_len: Base length for attention scaling.
            max_len: Maximum length for attention scaling. If equal to `base_len`, attention scaling is disabled.
            y_encoder_dim: Dimension of per-layer y embedding; `v_proj` input is `embed_dim` + `y_encoder_dim`.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_gate = nn.Linear(embed_dim, num_heads, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim + y_encoder_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_norm = LayerNorm(embed_dim)
        self.ff_norm = LayerNorm(embed_dim)
        self.ff = SwiGLU(embed_dim, ff_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Attention temperature stuff
        # If base_len == max_len, disable attention scaling (kappa = 0 ensures scale is always 1.0)
        self.disable_attention_scaling = base_len == max_len
        dk = float(self.head_dim)
        self.register_buffer("n0", torch.tensor(float(base_len)))
        self.register_buffer("max_len_f", torch.tensor(float(max_len)))
        if self.disable_attention_scaling:
            # Set kappa to 0 to ensure scaling is always 1.0
            self.register_buffer("kappa", torch.tensor(0.0))
        else:
            self.register_buffer(
                "kappa",
                torch.tensor((math.sqrt(dk) - 1.0) / math.log(max_len / base_len)),
            )

        # Zero-init output projections so residual branch is identity at start (attn=0, ff=0)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.q_gate.weight)
        nn.init.zeros_(self.ff.down.weight)

    def get_scale_param(self, eval_pos: int, *, device, dtype) -> torch.Tensor:
        if self.disable_attention_scaling:
            return torch.tensor(1.0, device=device, dtype=dtype)
        n = torch.as_tensor(eval_pos, device=device, dtype=dtype)
        n = torch.minimum(n, self.max_len_f.to(dtype))
        beta = 1.0 + self.kappa.to(dtype) * torch.clamp(torch.log(n / self.n0.to(dtype)), min=0.0)
        return beta

    def forward(self, x: torch.Tensor, y: torch.Tensor, eval_pos: int) -> torch.Tensor:
        """Compute the residual for this layer.

        Args:
            x: Input tensor of shape (L, B, D).
            y: Target embedding for the context region of shape (`eval_pos`, B, `y_encoder_dim`).
            eval_pos: Number of context positions (length of y and K/V).

        Returns:
            torch.tensor: Residual tensor to be added to input (same shape as input).
        """
        # switch to (B, L, D) for attention computation
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        B, L, _ = x.size()

        h = self.attn_norm(x)
        q = self.q_proj(h)
        gate = torch.sigmoid(self.q_gate(q))
        k = self.k_proj(h[:, :eval_pos])

        h_ctx = h[:, :eval_pos]
        v_in = torch.cat([h_ctx, y], dim=-1)
        v = self.v_proj(v_in)

        # reshape: (B, L, D) -> (B, num_heads, _, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        beta = self.get_scale_param(eval_pos, device=q.device, dtype=q.dtype)
        default_scale = 1.0 / math.sqrt(self.head_dim)
        custom_scale = default_scale * beta

        attn = F.scaled_dot_product_attention(q, k, v, scale=custom_scale).transpose(1, 2)
        attn = attn * gate.unsqueeze(-1)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        residual = attn + self.ff(self.ff_norm(x + attn))

        # back to (L, B, D) for output
        return residual.transpose(0, 1)
