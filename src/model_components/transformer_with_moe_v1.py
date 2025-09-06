# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 19:29
@Author : Xiaoguang Liang
@File : transformer.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
"""
Implementation of time conditioned Transformer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_moe_pytorch import SoftMoE, DynamicSlotsSoftMoE
from src.model_components.swiglu_ffn import SwiGLUFFN


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Input:
            x: [B,N,D]
        """
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        assert ctx.dim() == x.dim()
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class TimeMLP(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_h,
            dim_out,
            dim_ctx=None,
            act=F.relu,
            dropout=0.0,
            use_time=False,
    ):
        super().__init__()
        self.act = act
        self.use_time = use_time

        dim_h = int(dim_h)
        if use_time:
            self.fc1 = ConcatSquashLinear(dim_in, dim_h, dim_ctx)
            self.fc2 = ConcatSquashLinear(dim_h, dim_out, dim_ctx)
        else:
            self.fc1 = nn.Linear(dim_in, dim_h)
            self.fc2 = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        if self.use_time:
            x = self.fc1(x=x, ctx=ctx)
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.dropout(x)
        if self.use_time:
            x = self.fc2(x=x, ctx=ctx)
        else:
            x = self.fc2(x)

        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x,
            y=None,
            mask=None,
            alpha=None,
    ):
        y = y if y is not None else x
        b_a, n, c = x.shape
        b, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(
            b_a, n, self.num_heads, c // self.num_heads
        )
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        if alpha is not None:
            out, attention = self.forward_interpolation(
                queries, keys, values, alpha, mask
            )
        else:
            attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
            attention = attention.softmax(dim=2)
            attention = self.dropout(attention)
            out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TimeTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_self,
            dim_ctx=None,
            num_heads=1,
            mlp_ratio=2.0,
            act=F.leaky_relu,
            dropout=0.0,
            use_time=True,
            use_moe=True,
            num_experts=2,
            use_swiglu=False,
    ):
        super().__init__()
        self.use_time = use_time
        self.act = act
        self.attn = MultiHeadAttention(dim_self, dim_self, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim_self)

        mlp_ratio = int(mlp_ratio)

        # Initialize MLP layer
        mlp_hidden_dim = int(dim_self * mlp_ratio)

        self.use_swiglu = use_swiglu
        self.use_moe = use_moe
        if self.use_moe:
            print("using moe")
            self.moe = SoftMoE(
                dim=dim_self,
                seq_len=1024,
                num_experts=num_experts,
            )
            # self.moe = DynamicSlotsSoftMoE(
            #     dim=dim_self,  # model dimensions
            #     num_experts=num_experts,  # number of experts
            #     geglu=True
            # )
        elif use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(dim_self, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = TimeMLP(
                dim_self, dim_self * mlp_ratio, dim_self, dim_ctx, use_time=use_time
            )
        self.norm = nn.LayerNorm(dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        res = x
        x, attn = self.attn(x)
        x = self.attn_norm(x + res)

        res = x

        if self.use_moe:
            x = x + self.moe(x)
        elif self.use_swiglu:
            x = x + self.mlp(x)
        else:
            x = self.mlp(x, ctx=ctx)
        x = self.norm(x + res)

        return x, attn


class TimeTransformerDecoderLayer(TimeTransformerEncoderLayer):
    def __init__(
            self,
            dim_self,
            dim_ref,
            dim_ctx=None,
            num_heads=1,
            mlp_ratio=2,
            act=F.leaky_relu,
            dropout=0.0,
            use_time=True,
            use_moe=True,
            num_experts=2,
            use_swiglu=False,
    ):
        super().__init__(
            dim_self=dim_self,
            dim_ctx=dim_ctx,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act=act,
            dropout=dropout,
            use_time=use_time,
            use_moe=use_moe,
            num_experts=num_experts,
            use_swiglu=use_swiglu,
        )
        self.cross_attn = MultiHeadAttention(dim_self, dim_ref, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(dim_self)

    def forward(self, x, y, ctx=None):
        res = x
        x, attn = self.attn(x)
        x = self.attn_norm(x + res)

        res = x
        x, attn = self.cross_attn(x, y)
        x = self.cross_attn_norm(x + res)

        res = x

        if self.use_moe:
            x = x + self.moe(x)
        elif self.use_swiglu:
            x = x + self.mlp(x)
        else:
            x = self.mlp(x, ctx=ctx)
        x = self.norm(x + res)

        return x, attn


class TimeTransformerEncoder(nn.Module):
    def __init__(
            self,
            dim_self,
            dim_ctx=None,
            num_heads=1,
            mlp_ratio=2.0,
            act=F.leaky_relu,
            dropout=0.0,
            use_time=True,
            num_layers=3,
            last_fc=False,
            last_fc_dim_out=None,
            use_moe=True,
            num_moe_layers=2,
            num_experts=2,
            use_swiglu=False,
    ):
        super().__init__()
        self.last_fc = last_fc
        if last_fc:
            self.fc = nn.Linear(dim_self, last_fc_dim_out)
        self.layers = nn.ModuleList(
            [
                TimeTransformerEncoderLayer(
                    dim_self,
                    dim_ctx=dim_ctx,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    dropout=dropout,
                    use_time=use_time,
                    use_moe=True if num_layers - layer <= num_moe_layers else False,
                    num_experts=num_experts,
                    use_swiglu=use_swiglu,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, x, ctx=None):
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, ctx=ctx)

        if self.last_fc:
            x = self.fc(x)
        return x


class TimeTransformerDecoder(nn.Module):
    def __init__(
            self,
            dim_self,
            dim_ref,
            dim_ctx=None,
            num_heads=1,
            mlp_ratio=2.0,
            act=F.leaky_relu,
            dropout=0.0,
            use_time=True,
            num_layers=3,
            last_fc=True,
            last_fc_dim_out=None,
            num_moe_layers=2,
            num_experts=2,
            use_swiglu=False,
    ):
        super().__init__()
        self.last_fc = last_fc
        if last_fc:
            self.fc = nn.Linear(dim_self, last_fc_dim_out)

        self.layers = nn.ModuleList(
            [
                TimeTransformerDecoderLayer(
                    dim_self,
                    dim_ref,
                    dim_ctx=dim_ctx,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    dropout=dropout,
                    use_time=use_time,
                    use_moe=True if num_layers - layer <= num_moe_layers else False,
                    num_experts=num_experts,
                    use_swiglu=use_swiglu,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, x, y, ctx=None):
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, y=y, ctx=ctx)
        if self.last_fc:
            x = self.fc(x)

        return x
