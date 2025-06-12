import random
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from model_ulit.MTCN import ModernTCNBlock

########################################################################################
class Classifier(nn.Module):
    def __init__(self,config, type_n):
        super(Classifier, self).__init__()
        self.d_model = int(config["d_model"])
        self.Trans = nn.TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=self.d_model * 2, dropout=0.1,batch_first=True
                                                ,activation = 'gelu')
        self.mix_all = nn.Sequential(
            nn.Linear(self.d_model, type_n), nn.Sigmoid())
        self.mix1 = nn.Sequential(
            nn.Linear(self.d_model // 2, type_n), nn.Sigmoid())
        self.mix2 = nn.Sequential(
            nn.Linear(self.d_model // 2, type_n), nn.Sigmoid())
        self.predict_mix = multiplication(type_n)

    def forward(self, x):
        x = self.Trans(x)
        predict = self.mix_all(x)
        predict1 = self.mix1(x[:, :, :self.d_model // 2])
        predict2 = self.mix2(x[:, :, self.d_model // 2:])
        predict = self.predict_mix(predict, predict1, predict2)
        return predict


class FeatureExtractor(nn.Module):
    def __init__(self, n_samples, sfre, n_channels, config, type_n):
        super(FeatureExtractor, self).__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.sfre = sfre
        self.type_n = type_n
        self.C2D_seg_len = config["C2D_seg_len"]
        self.C1D_seg_len = config["C1D_seg_len"]
        self.d_model = int(config["d_model"])
        self.decdoer = C2D_decode_layers(n_channels, n_samples, self.C2D_seg_len, config["num_layers"], self.d_model)
        self.decdoer2 = C1D_decoder(self.d_model, n_samples, n_channels, self.C1D_seg_len)



    def forward(self, x,p1):
        x1 = self.decdoer(x,p1)
        x2 = self.decdoer2(x,p1)

        x = torch.cat([x1, x2], dim=1)
        return x


class multiplication(nn.Module):
    def __init__(self, type_n):
        super(multiplication, self).__init__()
        self.type_n = type_n
        self.parameters0 = nn.Parameter(torch.ones(type_n), requires_grad=True)
        self.parameters1 = nn.Parameter(torch.rand(type_n), requires_grad=True)
        self.parameters2 = nn.Parameter(torch.rand(type_n), requires_grad=True)

    def forward(self, x, x1, x2):
        x = (x * self.parameters0 + x1 * self.parameters1 + x2 * self.parameters2)
        p = (self.parameters0 + self.parameters1 + self.parameters2)
        return x/p


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
        self.ModerTCN = nn.Sequential(ModernTCNBlock(n_channels, (d_model // 4) * len(n_len), 9, 1))
        self.ModerTCN_p = Prompt_encode(d_model)
        self.Transformer = Transformer(d_model, len(n_len), n_channels)

    def forward(self, x, p1):
        for i in range(len(self.n_len)):
            # 给x加上一个维度，作为通道
            xi = self.U_2d[i](x)
            if i == 0:
                x_out = xi
            else:
                x_out = torch.cat([x_out, xi], dim=1)
        x = self.ModerTCN(x_out)
        p1 = self.ModerTCN_p(p1)
        x = torch.cat([x,p1],dim= 1)
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
            nn.Conv2d(n_channels, (d_model // 8) * n_channels, kernel_size=(3, 9), stride=(1, 1),
                      padding="same", groups=n_channels),
            nn.GroupNorm(n_channels, (d_model // 8) * n_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d((d_model // 8) * n_channels, (d_model // 4) * n_channels, kernel_size=(3, 9), stride=(1, 1),
                      padding="same", groups=n_channels),
            nn.GroupNorm(n_channels, (d_model // 4) * n_channels), nn.ReLU(),
        )
        self.ModerTCN = nn.Sequential(
                                      ModernTCNBlock(n_channels, d_model // 4, 25, 1),
                                      )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = rearrange(x, 'b l n (d c) -> b l c (n d)', c=self.n_len)
        x = self.resnet(x)
        x = rearrange(x, 'b d c n -> b d (n c)')
        x = self.ModerTCN(x)
        x = F.adaptive_avg_pool1d(x, self.d_model)
        return x


##########################################################################################
class C1D_decoder(nn.Module):
    def __init__(self, d_model, n_sample, n_channels, n_len):
        super(C1D_decoder, self).__init__()
        self.d_model = d_model
        self.n_sample = n_sample
        self.n_channels = n_channels
        self.n_len = n_len
        self.encoder = encoder(d_model, n_channels, self.n_len)
        self.ModerTCN_p = Prompt_encode(d_model)
        self.Transformer = Transformer(d_model, len(n_len), n_channels)

    def forward(self, x,p1):
        x = self.encoder(x)
        p1 = self.ModerTCN_p(p1)
        x = torch.cat([x,p1],dim=1)
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
                nn.Conv1d(n_channels, (d_model // 2) * n_channels, kernel_size=kernel_size, stride=seg_len[i],
                          padding=kernel_size // 2, groups=n_channels),
                nn.GroupNorm(n_channels, (d_model // 2) * n_channels), nn.GELU(),
                nn.Conv1d((d_model // 2) * n_channels, (d_model // 4) * n_channels, kernel_size=kernel_size, stride=1
                          , padding=kernel_size // 2, groups=n_channels),
                nn.GroupNorm(n_channels,(d_model // 4) * n_channels), nn.GELU(),
                ModernTCNBlock(n_channels, d_model // 4, 25, 1),
            ))
        self.encoder_len = len(seg_len)
        self.Model = nn.Sequential(ModernTCNBlock(n_channels, (d_model // 4) * len(seg_len), 9, 1))


    def forward(self, x):
        for i in range(self.encoder_len):
            x_i = self.encoder[i](x)
            x_i = F.adaptive_avg_pool1d(x_i, self.d_model)
            if i == 0:
                x_out = x_i
            else:
                x_out = torch.cat((x_out, x_i), dim=1)
        x = self.Model(x_out)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_lens, n_channels):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, d_model // 4, dropout=0.5, batch_first=True)
        self.norm = nn.BatchNorm1d((d_model // 4) * n_lens * n_channels + 1)
        self.mix = nn.Sequential(nn.Conv1d((d_model // 4) * n_lens * n_channels + 1, d_model // 2, 1, 1, 0),
                                 nn.BatchNorm1d(d_model // 2),
                                 nn.Dropout(0.5),
                                 nn.Linear(d_model, 1),
                                 nn.BatchNorm1d(d_model // 2), nn.Sigmoid(),
                                 )

    def forward(self, x):
        x = self.norm(self.attention(x, x, x, need_weights=False)[0] + x)
        x = self.mix(x).squeeze(2)
        return x



