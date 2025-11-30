from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU, LayerNorm, Linear

from .utils import clip_outliers, flash_context, normalize_data, pad_x

from enum import Enum
from typing import Union
import math
import random

import numpy as np
import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
from torch.nn.attention.flex_attention import create_block_mask


class Task(Enum):
    REG = 1
    CLS = 2


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data


def clip_outliers(data, eval_pos, n_sigma=4):
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

def ckpt(layer, *args):
    return checkpoint.checkpoint(layer, *args,
                      use_reentrant=False,          # <-- key change
                      preserve_rng_state=False)     # avoid extra syncs

class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        nbins: int = 1,
    ):
        super().__init__()
        self.n_out = n_out
        self.ninp = ninp
        self.nbins = nbins
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp, bias=True)
        self.head = nn.Sequential(nn.Linear(ninp, nhid, bias=False), nn.GELU(), nn.Linear(nhid, n_out + nbins, bias=False))
        self.xnorm = nn.LayerNorm(ninp, bias=False)
        self.ynorm = nn.LayerNorm(ninp, bias=False)
        # self._init_weights()
        # self.xynorm =nn.LayerNorm(ninp)

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        valid_mask: torch.Tensor = None,
        task: Literal["cls", "reg"] | Task | None = None,
        return_log_act_norms: bool = False,
    ) -> torch.Tensor:
        context_length = y_src.shape[0]
        B = y_src.shape[1]
        # x_src = clip_outliers(x_src, -1 if self.training  else context_length, n_sigma=4)
        x_src = normalize_data(x_src, -1 if self.training else context_length)
        x_src = clip_outliers(x_src, -1 if self.training  else context_length, n_sigma=10)
        x_src = torch.nan_to_num(x_src, nan=0)


        x_src = self.xnorm(self.encoder(x_src))
        if y_src.dim() == 2:
            y_src = y_src.unsqueeze(-1)
        y_src = self.ynorm(self.y_encoder(y_src))
        train_x = x_src[:context_length] + y_src
        src = torch.cat([train_x, x_src[context_length:]], 0)
        # mean = (src**2).mean(dim=-1, keepdim=True)
        # rms = torch.sqrt(mean)
        # src = src / (rms + 1e-6)
        # concat xy and rms after worked well for cls


        for l, layer in enumerate(self.transformer_encoder):
            src = layer(src, context_length)

        pred = self.head(src)

        pred = pred[context_length:]
        if task in ("cls", Task.CLS):
            pred = pred[..., : self.n_out]
        elif task in ("reg", Task.REG):
            pred = pred[..., -self.nbins :]
        return pred

    def predict(
        self,
        device: str,
        temperature: float,
        test_x: Union[torch.Tensor, np.ndarray],
        train_x: Union[torch.Tensor, np.ndarray],
        train_y: Union[torch.Tensor, np.ndarray],
        task: Task,
    ):
        to_numpy = not torch.is_tensor(train_x)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(train_x).to(device).float(),
            convert_to_torch_tensor(train_y).to(device).float(),
            convert_to_torch_tensor(test_x).to(device).float(),
        )
        dim = train_x.dim()
        if dim == 2:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        else:
            assert False, "Predict only supports one dataset at a time"

        num_classes = len(torch.unique(train_y))
        full_x = pad_x(torch.cat((train_x, test_x), dim=0), self.num_features)

        # forward
        output = self.forward(x_src=full_x, y_src=train_y, task=task)
        output = output[..., :num_classes] / temperature
        output = torch.nn.functional.softmax(output, dim=-1)
        if dim == 2:
            output.squeeze_(1)
        if to_numpy:
            output = output.detach().cpu().numpy()
        return output

    @classmethod
    def load(cls, model_state, config):
        assert config.model.max_num_classes > 2
        nhid_factor = getattr(config.model, "nhid_factor", None)
        nhid = getattr(config.model, "ff_dim", None)
        if nhid_factor is not None:
            nhid = config.model.emsize * nhid_factor
        if nhid is None:
            nhid = config.model.emsize * 4  # sensible default if neither field is present
        nbins = getattr(config.model, "nbins", 1)
        model = TabDPTModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=nhid,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            nbins=nbins,
        )

        module_prefix = "_orig_mod."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}

        # Only load parameters that both exist and match shape; skip the rest to allow older/newer checkpoints.
        current_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in model_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append(k)

        if skipped:
            print(f"[TabDPTModel] Skipping {len(skipped)} unmatched keys when loading checkpoint.")

        model.load_state_dict(filtered_state, strict=False)
        model.to(config.env.device)
        model.eval()
        return model



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # LayerNorm might be faster if not using compile because it has fused kernels
        # while RMSNorm doesn't
        # norm = lambda x: RMSNorm(x)
        norm = lambda x: RMSNorm(x, bias=False)
        self.attn_norm = norm(embed_dim)
        self.ff_norm = norm(embed_dim)
        # self.ff = nn.Sequential(Linear(embed_dim, ff_dim, bias=False), GELU(), Linear(ff_dim, embed_dim, bias=False))
        self.q_norm = norm(self.head_dim)
        self.k_norm = norm(self.head_dim)

    def forward(self, x, context_length):
        x = x.transpose(0, 1)
        B, L, _ = x.size()
        h = self.attn_norm(x)
        q, ff_h, ff_gate = self.q_proj(h).chunk(3, dim=-1)
        k, v = self.kv_proj(h[:, :context_length]).chunk(2, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q*math.log2(context_length)/10
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = attn.reshape(B, L, self.num_heads * self.head_dim)
        ff_x = ff_h*F.silu(ff_gate)
        # attn_ff = torch.cat([attn, ff_h], dim=-1)
        residual = self.ff_norm(self.out_proj(attn+ff_x))
        x_out = x + residual
        return x_out.transpose(0, 1)
