"""Per-trace 1D U-Net with inter-trace self-attention."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


class _DoubleConv1d(nn.Module):
    """(Conv1d->BN->ReLU) x 2 with same temporal length."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _TraceSelfAttention(nn.Module):
    """Apply self-attention along the trace dimension at each time sample."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"attention channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        hidden = channels * int(ffn_multiplier)
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, n_traces: int) -> torch.Tensor:
        bt, channels, n_time = x.shape
        if bt % n_traces != 0:
            raise ValueError(f"Batch*trace dimension {bt} is not divisible by n_traces {n_traces}.")
        batch = bt // n_traces

        tokens = x.view(batch, n_traces, channels, n_time)
        tokens = tokens.permute(0, 3, 1, 2).reshape(batch * n_time, n_traces, channels)
        attn_in = self.norm_attn(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm_ffn(tokens))

        tokens = tokens.view(batch, n_time, n_traces, channels).permute(0, 2, 3, 1)
        return tokens.reshape(bt, channels, n_time)


@register_model("trace_attention_unet")
class TraceAttentionUNet(nn.Module):
    """Shared single-trace 1D U-Net plus bottleneck attention between traces.

    Input/Output tensors use the repository convention ``(B, C, trace, time)``.
    The same 1D U-Net is applied to every trace independently, then multi-head
    self-attention mixes bottleneck features across traces for each compressed
    time sample.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"TraceAttentionUNet depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev = in_channels
        for c in chans:
            self.encoders.append(_DoubleConv1d(prev, c))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev = c

        bottleneck_ch = chans[-1] * 2
        self.bottleneck = _DoubleConv1d(chans[-1], bottleneck_ch)
        self.trace_attention = _TraceSelfAttention(
            channels=bottleneck_ch,
            num_heads=attention_heads,
            dropout=attention_dropout,
            ffn_multiplier=ffn_multiplier,
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = bottleneck_ch
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose1d(dec_in, c, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv1d(c * 2, c))
            dec_in = c

        self.head = nn.Conv1d(chans[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, trace, time), got {tuple(x.shape)}.")
        batch, channels, n_traces, n_time = x.shape
        h = x.permute(0, 2, 1, 3).reshape(batch * n_traces, channels, n_time)

        skips: List[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)
        h = self.trace_attention(h, n_traces=n_traces)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        h = self.head(h)
        return h.view(batch, n_traces, -1, n_time).permute(0, 2, 1, 3)
