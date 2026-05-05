import copy
from inspect import isfunction
from typing import Callable, List, Optional

from einops import rearrange
import torch
from torch import nn


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(
            data_dim
        )  # TODO: Check if elementwise_affine=True is correct
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, ..., data_dim)
        # t: (batch, norm_cond_dim)
        # return: (batch, data_dim)
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)

        # Add singleton dimensions to alpha and beta
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])

        return x * (1 + alpha) + beta


class SequentialCond(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)):
                # print(f'Passing on args to {module}', [a.shape for a in args])
                input = module(input, *args, **kwargs)
            else:
                # print(f'Skipping passing args to {module}', [a.shape for a in args])
                input = module(input)
        return input


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


def linear_norm_activ_dropout(
    input_dim: int,
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    return SequentialCond(*layers)


def create_simple_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            linear_norm_activ_dropout(
                prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            )
        )
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim, bias=bias))
    return SequentialCond(*layers)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim} is not implemented"
            )

        layers = []
        prev_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(
                linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
                )
            )
            prev_dim = hidden_dim
        self.model = SequentialCond(*layers)
        self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        num_blocks: int = 1,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model = SequentialCond(
            linear_norm_activ_dropout(
                input_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            ),
            *[
                ResidualMLPBlock(
                    hidden_dim,
                    hidden_dim,
                    num_hidden_layers,
                    hidden_dim,
                    activation,
                    bias,
                    norm,
                    dropout,
                    norm_cond_dim,
                )
                for _ in range(num_blocks)
            ],
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x, *args, **kwargs)


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(
            N, -1
        )  # (N, D * 2 * num_frequencies + D)
        return embedded




class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None, context_mask=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if exists(context_mask):
            if context_mask.ndim == 2:
                mask = context_mask[:, None, None, :].to(torch.bool)
            elif context_mask.ndim == 3:
                mask = context_mask[:, None, :, :].to(torch.bool)
            else:
                raise ValueError(
                    f"context_mask must have shape (B, K) or (B, Q, K), got {context_mask.shape}"
                )
            dots = dots.masked_fill(mask, torch.finfo(dots.dtype).min)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x, context, context_mask=None):
        for i, (sa, ca, ff) in enumerate(self.layers):
            x = sa(x) + x
            x = ca(x, context=context, context_mask=context_mask) + x
            x = ff(x) + x
        return x


class DecoderLayerDoubleCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        max_temporal_len: int = 32,
    ):
        super().__init__()
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, max_temporal_len, dim) * 0.02
        )
        self.temporal_alpha = nn.Parameter(torch.tensor(1.0))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            t_sa = SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca1 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca2 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)

            mods = [
                PreNorm(dim, sa,  norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, t_sa, norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, ca1, norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, ca2, norm=norm, norm_cond_dim=norm_cond_dim),
                PreNorm(dim, ff,  norm=norm, norm_cond_dim=norm_cond_dim),
            ]
            self.layers.append(nn.ModuleList([m for m in mods if m is not None]))

    def _temporal_sa(self, x, t_sa, batch_size, seq_len):
        """
        x: (B*T, N, D)  →  reshape to (B*N, T, D)  →  SA  →  reshape back
        """
        BT, N, D = x.shape
        # add temporal positional embedding
        pos = self.temporal_pos_embedding[:, :seq_len]          # (1, T, D)
        x_t = x.reshape(batch_size, seq_len, N, D)
        x_t = x_t + pos.unsqueeze(2)                            # broadcast over N
        x_t = x_t.permute(0, 2, 1, 3).reshape(batch_size * N, seq_len, D)

        residual = x_t
        x_t = t_sa(x_t)
        x_t = residual + self.temporal_alpha * x_t

        x_t = x_t.reshape(batch_size, N, seq_len, D).permute(0, 2, 1, 3)
        return x_t.reshape(BT, N, D)

    def forward(self, x, context1, context2, context1_mask=None, context2_mask=None,batch_size=None,seq_len=None):
        # layers list: [sa_prenorm, t_sa_prenorm, ca1_prenorm, ca2_prenorm, ff_prenorm]
        for sa, t_sa, ca1, ca2, ff in self.layers:
            x = sa(x) + x
            x = self._temporal_sa(x, t_sa, batch_size, seq_len)
            x = ca1(x, context=context1, context_mask=context1_mask) + x
            x = ca2(x, context=context2, context_mask=context2_mask) + x
            x = ff(x) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = DecoderLayer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, x, context, context_mask=None):
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(
            x,
            context=context,
            context_mask=context_mask,
        )
        return x


class DecoderDoubleCrossAttn(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        max_temporal_len: int = 32,
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = DecoderLayerDoubleCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            max_temporal_len=max_temporal_len,
        )

    def forward(self, x, context1, context2, context1_mask=None, context2_mask=None,batch_size=None,seq_len=None):
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(
            x,
            context1=context1,
            context2=context2,
            context1_mask=context1_mask,
            context2_mask=context2_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        return x
