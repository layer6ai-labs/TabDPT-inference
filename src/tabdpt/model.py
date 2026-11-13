from typing import Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear

from .utils import (
    clip_outliers,
    convert_to_torch_tensor,
    flash_context,
    maskmean,
    maskstd,
    normalize_data,
)


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

    @flash_context
    def encode_context(
        self,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        is_cls: bool = True,
    ) -> tuple[list, int, dict]:
        """Precomputes and caches the Key and Value states across all transformer layers
        for the given context dataset.

        Args:
            x_ctx: Context features with dimensions (B, n_ctx, F) or (n_ctx, F).
            y_ctx: Context targets with dimensions (B, n_ctx) or (n_ctx,).
            is_cls: True for classification, False for regression.

        Returns:
            tuple: (kv_cache, n_ctx, stats)
                kv_cache: List of (k, v) tuples for each layer in self.transformer_encoder.
                n_ctx: Number of context rows.
                stats: Dictionary of preprocessing normalization and outlier clipping statistics.
        """
        device = next(self.parameters()).device
        dtype = self.encoder.weight.dtype

        x_ctx = convert_to_torch_tensor(x_ctx).to(device=device, dtype=dtype)
        y_ctx = convert_to_torch_tensor(y_ctx).to(device=device)

        if x_ctx.ndim == 2:
            x_ctx = x_ctx.unsqueeze(0)
        if y_ctx.ndim == 1:
            y_ctx = y_ctx.unsqueeze(0)

        x_ctx = x_ctx.transpose(0, 1)  # (n_ctx, B, F)
        y_ctx = y_ctx.transpose(0, 1)  # (n_ctx, B)
        n_ctx = y_ctx.shape[0]
        n_think = self.n_thinking_rows

        # Preprocess context data and record normalization & outlier clipping stats
        # Step 1: clip outliers
        mask1 = ~torch.isnan(x_ctx)
        mean1 = maskmean(x_ctx, mask1, dim=0)
        std1 = maskstd(x_ctx, mask1, dim=0)
        cutoff1 = self.clip_sigma * std1
        mask1 &= cutoff1 >= torch.abs(x_ctx - mean1)
        cutoff1 = self.clip_sigma * maskstd(x_ctx, mask1, dim=0)
        c1_min = mean1 - cutoff1
        c1_max = mean1 + cutoff1
        x_ctx = torch.clip(x_ctx, c1_min, c1_max)

        # Step 2: normalize data
        mask2 = ~torch.isnan(x_ctx)
        mean2 = maskmean(x_ctx, mask2, dim=0)
        std2 = maskstd(x_ctx, mask2, dim=0) + 1e-6
        x_ctx = (x_ctx - mean2) / std2

        # Step 3: clip outliers after normalization
        mask3 = ~torch.isnan(x_ctx)
        mean3 = maskmean(x_ctx, mask3, dim=0)
        cutoff3 = self.clip_sigma * maskstd(x_ctx, mask3, dim=0)
        mask3 &= cutoff3 >= torch.abs(x_ctx - mean3)
        cutoff3 = self.clip_sigma * maskstd(x_ctx, mask3, dim=0)
        c3_min = mean3 - cutoff3
        c3_max = mean3 + cutoff3
        x_ctx = torch.clip(x_ctx, c3_min, c3_max)
        x_ctx = torch.nan_to_num(x_ctx, nan=0.0, posinf=0.0, neginf=0.0)

        stats = {
            "c1_min": c1_min,
            "c1_max": c1_max,
            "norm_mean": mean2,
            "norm_std": std2,
            "c3_min": c3_min,
            "c3_max": c3_max,
        }

        x_ctx = self.encoder(x_ctx)
        src = self.enc_norm(x_ctx)

        if n_think > 0:
            B = src.shape[1]
            src = torch.cat([self.thinking_embed.unsqueeze(1).expand(n_think, B, -1), src], dim=0)

        eval_pos = n_ctx + n_think
        kv_cache = []

        for l, layer in enumerate(self.transformer_encoder):
            if is_cls:
                y_cls = y_ctx.long().clamp(min=0, max=self.n_out - 1)
                y_emb = self.cls_y_encoders[l](y_cls)
            else:
                y_emb = self.reg_y_encoders[l](y_ctx.unsqueeze(-1))
            if n_think > 0:
                B = y_emb.shape[1]
                y_emb = torch.cat([y_emb.new_zeros(n_think, B, y_emb.shape[-1]), y_emb], dim=0)

            src, (k, v) = layer.encode_context(src, y_emb, eval_pos)
            kv_cache.append((k, v))

        return kv_cache, n_ctx, stats

    @flash_context
    def predict_with_cache(
        self,
        x_qry: torch.Tensor,
        kv_cache: list | tuple,
        n_ctx: int | None = None,
        stats: dict | tuple | None = None,
        is_cls: bool = True,
    ) -> torch.Tensor:
        """Runs fast inference on query samples using precomputed context KV cache.

        Args:
            x_qry: Query features with dimensions (B, n_qry, F) or (n_qry, F).
            kv_cache: Precomputed list of (k, v) tuples per layer (or tuple of (kv_cache, n_ctx, stats)).
            n_ctx: Context size (optional if kv_cache passed as unpacked 3-tuple).
            stats: Precomputed normalization & outlier clipping statistics dict.
            is_cls: Kept for API compatibility.

        Returns:
            torch.Tensor: Predicted values with dimensions (n_qry, B, O).
        """
        if isinstance(kv_cache, tuple) and len(kv_cache) == 3 and n_ctx is None and stats is None:
            kv_cache, n_ctx, stats = kv_cache

        device = next(self.parameters()).device
        dtype = self.encoder.weight.dtype

        x_qry = convert_to_torch_tensor(x_qry).to(device=device, dtype=dtype)
        if x_qry.ndim == 2:
            x_qry = x_qry.unsqueeze(1)
        elif x_qry.ndim == 3:
            x_qry = x_qry.transpose(0, 1)

        eval_pos = n_ctx + self.n_thinking_rows

        # Preprocess query features using context stats (supports dict, tuple, or None)
        if stats is not None:
            c1_min = c1_max = norm_mean = norm_std = c3_min = c3_max = None
            if isinstance(stats, dict):
                c1_min = stats.get("c1_min", stats.get("clip1_min"))
                c1_max = stats.get("c1_max", stats.get("clip1_max"))
                norm_mean = stats.get("norm_mean", stats.get("mean2"))
                norm_std = stats.get("norm_std", stats.get("std2"))
                c3_min = stats.get("c3_min", stats.get("clip3_min"))
                c3_max = stats.get("c3_max", stats.get("clip3_max"))
            elif isinstance(stats, (tuple, list)):
                if len(stats) == 6:
                    c1_min, c1_max, norm_mean, norm_std, c3_min, c3_max = stats
                elif len(stats) == 4:
                    c1_min, c1_max, norm_mean, norm_std = stats
                elif len(stats) == 2:
                    norm_mean, norm_std = stats

            if c1_min is not None and c1_max is not None:
                c1_min = c1_min.to(device=device, dtype=dtype)
                c1_max = c1_max.to(device=device, dtype=dtype)
                x_qry = torch.clip(x_qry, c1_min, c1_max)

            if norm_mean is not None and norm_std is not None:
                norm_mean = norm_mean.to(device=device, dtype=dtype)
                norm_std = norm_std.to(device=device, dtype=dtype)
                x_qry = (x_qry - norm_mean) / norm_std

            if c3_min is not None and c3_max is not None:
                c3_min = c3_min.to(device=device, dtype=dtype)
                c3_max = c3_max.to(device=device, dtype=dtype)
                x_qry = torch.clip(x_qry, c3_min, c3_max)

            x_qry = torch.nan_to_num(x_qry, nan=0.0, posinf=0.0, neginf=0.0)

        x_qry = self.encoder(x_qry)
        src_qry = self.enc_norm(x_qry)

        for l, layer in enumerate(self.transformer_encoder):
            k, v = kv_cache[l]
            k = k.to(device=device, dtype=dtype)
            v = v.to(device=device, dtype=dtype)
            src_qry = layer.forward_query(src_qry, k, v, eval_pos)

        pred = self.head(src_qry.float())
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

    def encode_context(
        self, x: torch.Tensor, y: torch.Tensor, eval_pos: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Encode context tokens, compute and return normalized key and value representations,
        and return the updated context representations for the next layer.

        Args:
            x: Context tensor of shape (eval_pos, B, D).
            y: Target embedding for context region of shape (eval_pos, B, y_encoder_dim).
            eval_pos: Total context positions (length of y and K/V).

        Returns:
            tuple: (updated_x, (k, v)) where:
                updated_x: Residual-updated context tensor of shape (eval_pos, B, D).
                k: Key tensor of shape (B, num_heads, eval_pos, head_dim) normalized by k_norm.
                v: Value tensor of shape (B, num_heads, eval_pos, head_dim).
        """
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

        next_x = (x + residual).transpose(0, 1)
        return next_x, (k, v)

    def forward_query(
        self, x_qry: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eval_pos: int
    ) -> torch.Tensor:
        """Run attention on query tokens attending to cached context keys and values.

        Args:
            x_qry: Query tensor of shape (n_qry, B, D).
            k: Cached key tensor of shape (B_ctx, num_heads, eval_pos, head_dim).
            v: Cached value tensor of shape (B_ctx, num_heads, eval_pos, head_dim).
            eval_pos: Total context positions (for attention temperature scaling).

        Returns:
            torch.Tensor: Residual-updated query tensor of shape (n_qry, B, D).
        """
        x = x_qry.transpose(0, 1)
        B, L, _ = x.size()

        if k.shape[0] != B:
            k = k.expand(B, -1, -1, -1)
            v = v.expand(B, -1, -1, -1)

        h = self.attn_norm(x)
        q = self.q_proj(h)
        gate = torch.sigmoid(self.q_gate(q))
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        beta = self.get_scale_param(eval_pos, device=q.device, dtype=q.dtype)
        default_scale = 1.0 / math.sqrt(self.head_dim)
        custom_scale = default_scale * beta

        attn = F.scaled_dot_product_attention(q, k, v, scale=custom_scale).transpose(1, 2)
        attn = attn * gate.unsqueeze(-1)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        residual = attn + self.ff(self.ff_norm(x + attn))

        return (x + residual).transpose(0, 1)
