import random
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


########################################################################################


class NEWNet(nn.Module):
    def __init__(self, n_samples, sfre, n_channels, config, type_n):
        super(NEWNet, self).__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.sfre = sfre
        self.type_n = type_n
        self.C2D_seg_len = config["C2D_seg_len"]
        self.C1D_seg_len = config["C1D_seg_len"]
        self.d_model = int(config["d_model"])
        self.decdoer = C2D_decode_layers(n_channels, n_samples, self.C2D_seg_len, config["num_layers"], self.d_model)
        self.decdoer2 = C1D_decoder_layer(self.d_model, n_samples, n_channels, self.C1D_seg_len, config["num_retnet"])

        self.mix_all = nn.Sequential(
            nn.Linear(self.d_model, type_n), nn.Sigmoid())
        self.mix1 = nn.Sequential(
            nn.Linear(self.d_model // 2, type_n), nn.Sigmoid())
        self.mix2 = nn.Sequential(
            nn.Linear(self.d_model // 2, type_n), nn.Sigmoid())
        self.predict_mix = multiplication(type_n)

    def forward(self, x, model=None):
        x1 = self.decdoer(x)
        x2 = self.decdoer2(x)

        x = torch.cat([x1, x2], dim=1)
        predict = self.mix_all(x)
        predict1 = self.mix1(x1)
        predict2 = self.mix2(x2)
        predict = self.predict_mix(predict, predict1, predict2)
        if model == "train":
            return predict, x
        else:
            return predict


class multiplication(nn.Module):
    def __init__(self, type_n):
        super(multiplication, self).__init__()
        self.type_n = type_n
        self.parameters0 = nn.Parameter(torch.ones(type_n), requires_grad=True)
        self.parameters1 = nn.Parameter(torch.rand(type_n), requires_grad=True)
        self.parameters2 = nn.Parameter(torch.rand(type_n), requires_grad=True)

    def forward(self, x, x1, x2):
        x = (x * self.parameters0 + x1 * self.parameters1 + x2 * self.parameters2) / \
            (self.parameters0 + self.parameters1 + self.parameters2)
        return x


##########################################################################################
class C2D_decode_layers(nn.Module):
    def __init__(self, n_channels, n_sample, n_len, num_layers, d_model):
        super(C2D_decode_layers, self).__init__()
        self.n_channels = n_channels
        self.n_len = n_len
        self.n_sge_len = [n_sample / i for i in n_len]
        self.num_layers = num_layers
        self.U_2d = nn.ModuleList()
        for i in range(len(self.n_len)):
            self.U_2d.append(res_block(n_channels, int(self.n_len[i]), n_sample,
                                       num_layers, d_model))
        self.ModerTCN = ModernTCNBlock(n_channels, (d_model // 4) * len(n_len), 15, 1)
        self.Transformer = Transformer(d_model, len(n_len), n_channels)

    def forward(self, x):
        for i in range(len(self.n_len)):
            # 给x加上一个维度，作为通道
            xi = self.U_2d[i](x)
            if i == 0:
                x_out = xi
            else:
                x_out = torch.cat([x_out, xi], dim=1)
        x = self.ModerTCN(x_out)
        x = rearrange(x, '(b l) m n -> b (l m) n', l=self.n_channels)
        x = self.Transformer(x)
        return x


#########################################################################################
class res_block(nn.Module):
    def __init__(self, n_channels, n_len, n_sample, num_layers, d_model):
        super(res_block, self).__init__()
        self.n_channels = n_channels
        self.n_len = n_len
        self.num_layers = num_layers
        self.n_sample = n_sample
        self.d_model = d_model
        self.resnet = nn.Sequential(
            nn.Conv2d(1, d_model // 8, kernel_size=(5, 9), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(d_model // 8), nn.GELU(),
            nn.Conv2d(d_model // 8, d_model // 4, kernel_size=(3, 9), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(d_model // 4),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.ModerTCN = ModernTCNBlock(n_channels, d_model // 4, 25, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.unsqueeze(2)
        x = rearrange(x, 'b l m n (d c) -> (b l) m c (n d)', c=self.n_len)
        x = self.resnet(x)
        x = rearrange(x, 'b m c n -> b m (n c)')
        x = self.ModerTCN(x)
        x = F.adaptive_avg_pool1d(x, self.d_model)
        return x


##########################################################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        super(ResBlock, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding), nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                                             padding=padding), nn.BatchNorm2d(out_channels)
                                   )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.shortcut(x) + out
        return out


##########################################################################################
class C1D_decoder_layer(nn.Module):
    def __init__(self, d_model, n_sample, n_channels, seg_len, n_layers):
        super(C1D_decoder_layer, self).__init__()
        self.d_model = d_model
        self.n_sample = n_sample
        self.n_channels = n_channels
        self.seg_len = seg_len
        self.encoder = C1D_decoder(self.d_model, self.n_sample, self.n_channels, seg_len)

    def forward(self, x):
        x_out = self.encoder(x)
        return x_out


##########################################################################################
class C1D_decoder(nn.Module):
    def __init__(self, d_model, n_sample, n_channels, n_len):
        super(C1D_decoder, self).__init__()
        self.d_model = d_model
        self.n_sample = n_sample
        self.n_channels = n_channels
        self.n_len = n_len
        self.encoder = encoder(d_model, n_channels, self.n_len)
        self.Transformer = Transformer(d_model, len(n_len), n_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.Transformer(x)
        return x


class encoder(nn.Module):
    def __init__(self, d_model, n_channels, seg_len):
        super(encoder, self).__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.encoder = nn.ModuleList()
        for i in range(len(seg_len)):
            kernel_size = int(5 * seg_len[i])
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.encoder.append(nn.Sequential(
                nn.Conv1d(1, d_model // 2, kernel_size=kernel_size, stride=seg_len[i],
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(d_model // 2), nn.GELU(),
                nn.Conv1d(d_model // 2, d_model // 4, kernel_size=kernel_size, stride=1
                          , padding=kernel_size // 2),
                nn.BatchNorm1d(d_model // 4), nn.GELU(),
                ModernTCNBlock(n_channels, d_model // 4, 25, 1),
            ))
        self.encoder_len = len(seg_len)
        self.Model = ModernTCNBlock(n_channels, (d_model // 4) * len(seg_len), 15, 1)

    def forward(self, x):
        x = rearrange(x, 'b l n -> (b l) n')
        x = x.unsqueeze(1)
        for i in range(self.encoder_len):
            x_i = self.encoder[i](x)
            x_i = F.adaptive_avg_pool1d(x_i, self.d_model)
            if i == 0:
                x_out = x_i
            else:
                x_out = torch.cat((x_out, x_i), dim=1)
        x = self.Model(x_out)
        x = rearrange(x, '(b l) d n -> b (l d) n', l=self.n_channels)
        return x


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, groups_num=1):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        self.pw_con1 = nn.Sequential(nn.Conv1d(
            in_channels=M * D,
            out_channels=r * M * D,
            kernel_size=1,
            groups=groups_num
        ), nn.GELU(), nn.Dropout(0.2)
        )
        self.pw_con2 = nn.Sequential(nn.Conv1d(
            in_channels=r * M * D,
            out_channels=M * D,
            kernel_size=1,
            groups=groups_num
        ), nn.BatchNorm1d(M * D))

    def forward(self, x):
        # x: [B, M*D, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
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
        x = rearrange(x_emb, '(b m) d n -> b (m d) n', m=self.M)  # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x)  # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)  # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)  # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', m=self.M)  # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0, 2, 1, 3)  # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')  # [B, D, M, N] -> [B, D*M, N]

        x = self.conv_ffn2(x)  # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> (b m) d n', m=self.M)  # [B, D*M, N] -> [B, D, M, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]


class Transformer(nn.Module):
    def __init__(self, d_model, n_lens, n_channels):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, d_model // 4, dropout=0.5, batch_first=True)
        self.norm = nn.BatchNorm1d((d_model // 4) * n_lens * n_channels)
        self.mix = nn.Sequential(nn.Conv1d((d_model // 4) * n_lens * n_channels, d_model // 2, 1, 1, 0),
                                 nn.BatchNorm1d(d_model // 2),
                                 nn.Dropout(0.5),
                                 nn.Linear(d_model, 1),
                                 nn.BatchNorm1d(d_model // 2), nn.Sigmoid(),
                                 )

    def forward(self, x):
        x = self.norm(self.attention(x, x, x, need_weights=False)[0] + x)
        x = self.mix(x).squeeze(2)
        return x
