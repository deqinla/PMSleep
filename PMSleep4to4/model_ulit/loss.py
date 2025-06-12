import torch
import torch.nn as nn
import torch.nn.functional as F


def hot_y(y, type_n):
    y_hot = torch.zeros(y.shape[0], type_n).to(y.device)
    place = torch.arange(y.shape[0], dtype=torch.long)
    y = y.to(torch.long)
    y_hot[place, y] = 1
    return y_hot

class trainLoss2(nn.Module):
    def __init__(self, type_n, device):
        super(trainLoss2, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(device)
    def forward(self, output_h, target_h):
        loss = self.loss(output_h, target_h)
        return loss

class trainLoss1(nn.Module):
    def __init__(self, type_n, device):
        super(trainLoss1, self).__init__()
        self.CosLoss = CosLoss(type_n).to(device)
    def forward(self, target_h, x):
        cos_loss = self.CosLoss(x, target_h)

        return cos_loss


class LinearLoss(nn.Module):
    def __init__(self, type_n):
        super(LinearLoss, self).__init__()
        self.type_n = type_n

    def forward(self, x, y_real):
        x = x.reshape(x.shape[0], -1)
        y_real = torch.argmax(y_real, dim=1)
        # 检查预测的类别是否和真实的类别一致，不一致则删除不考虑
        center = torch.zeros((self.type_n, x.shape[1])).to(x.device)
        loss_postive = 0
        for i in range(self.type_n):
            # 取交集
            place = torch.where(y_real == i)[0]
            if len(x[place]) == 0:
                continue
            center[i] = (torch.sum(x[place], dim=0)) / len(x[place])
            loss_postive = (loss_postive +
                            torch.mean(F.pairwise_distance(x[place],
                                                           center[i].clone().unsqueeze(0).expand(len(x[place]), -1))))
        # 去除center中的0
        place = torch.where(torch.sum(center, dim=1) != 0)[0]
        center = center[place]
        # 计算类间的欧几里得距离
        loss_nagetive = 0
        real_n = len(center)
        for i in range(real_n):
            for j in range(i + 1, real_n):
                loss_nagetive = loss_nagetive + F.pairwise_distance(center[i].unsqueeze(0), center[j].unsqueeze(0))
        loss_nagetive = (loss_nagetive / (real_n * (real_n - 1) / 2)) if real_n != 1 else 0
        loss_postive = loss_postive / real_n if real_n != 0 else 1
        loss = loss_postive / (loss_nagetive + loss_postive)
        return loss


class CosLoss(nn.Module):
    def __init__(self, type_n):
        super(CosLoss, self).__init__()
        self.type_n = type_n

    def forward(self, x, y_real):
        x = x.reshape(x.shape[0], -1)
        y_real = torch.argmax(y_real, dim=1)
        # 检查预测的类别是否和真实的类别一致，不一致则删除不考虑
        center = torch.zeros((self.type_n, x.shape[1])).to(x.device)
        loss_postive = torch.zeros((self.type_n)).to(x.device)
        for i in range(self.type_n):
            # 取交集
            place = torch.where(y_real == i)[0]
            if len(x[place]) == 0:
                continue
            center[i] = (torch.sum(x[place], dim=0)) / len(x[place])
            # 计算类内与聚类中心的余弦相似度
            loss_postive[i] = torch.mean(
                F.cosine_similarity(x[place], center[i].clone().unsqueeze(0).expand(len(x[place]), -1)))
        # 去除center中的0
        place = torch.where(torch.sum(center, dim=1) != 0)[0]
        center = center[place]
        loss_postive = loss_postive[place]
        # 计算类间的余弦相似度
        loss_nagetive = 0
        real_n = len(center)
        for i in range(real_n):
            for j in range(i + 1, real_n):
                loss_nagetive = loss_nagetive + F.cosine_similarity(center[i].unsqueeze(0), center[j].unsqueeze(0))
        loss_nagetive = (loss_nagetive / (real_n * (real_n - 1) / 2)) ** 2 if real_n != 1 else 0
        loss_postive = torch.mean(loss_postive) ** 2
        loss = -torch.log((loss_postive  + 0.001)/ (loss_nagetive + 1.001))
        return loss


class evaLoss(nn.Module):
    def __init__(self, type_n):
        super(evaLoss, self).__init__()
        self.Criterion = nn.CrossEntropyLoss(reduction='mean')
        self.type_n = type_n

    def forward(self, output, target):
        loss = self.Criterion(output, target)
        return loss
