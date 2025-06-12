import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, groups_num=1):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        self.pw_con1 = nn.Sequential(nn.Conv1d(
            in_channels=M * D,
            out_channels=r * M * D,
            kernel_size=1,
            groups=groups_num
        ),  nn.GELU(),
        )
        self.pw_con2 = nn.Sequential(nn.Conv1d(
            in_channels=r * M * D,
            out_channels=M * D,
            kernel_size=1,
            groups=groups_num
        ), nn.BatchNorm1d(M * D), nn.GELU())

    def forward(self, x):
        # x: [B, M*D, N]
        x = self.pw_con2(self.pw_con1(x))
        return x  # x: [B, M*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=M * D,
            out_channels=M * D,
            kernel_size=kernel_size,
            groups=M * D,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(M * D)
        self.conv_ffn1 = ConvFFN(M, D, r, groups_num=M)
        self.conv_ffn2 = ConvFFN(M, D, r, groups_num=D)
        self.M = M
        self.D = D

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        x = self.dw_conv(x_emb)  # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)  # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)  # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', m=self.M)  # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0, 2, 1, 3)  # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')  # [B, D, M, N] -> [B, D*M, N]

        x = self.conv_ffn2(x)  # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b (m d) n', m=self.M)  # [B, D*M, N] -> [B, D, M, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]