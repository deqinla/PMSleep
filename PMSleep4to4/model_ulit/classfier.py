import torch
import torch.nn as nn
##########################################################################################
class Classfier_Linear(nn.Module):
    def __init__(self,Final_len,type_n):
        super(Classfier_Linear, self).__init__()
        self.FL = nn.Sequential(nn.Linear(int(Final_len), int(Final_len / 2)), nn.Dropout(0.5),
                                nn.Linear(int(Final_len / 2), 5), nn.Sigmoid())
        self.FL1 = nn.Sequential(nn.Linear(int(Final_len), int(Final_len / 2)), nn.Dropout(0.5),
                                 nn.Linear(int(Final_len / 2), 5),nn.Sigmoid())
        self.FL2 = nn.Sequential(nn.Linear(int(Final_len), int(Final_len / 2)), nn.Dropout(0.5),
                                 nn.Linear(int(Final_len / 2), 5), nn.Sigmoid())
        self.mix = multiplication(type_n)
    def forward(self, x,x1,x2):
        predict = self.FL(x)
        predict1 = self.FL1(x1)
        predict2 = self.FL2(x2)
        predict = self.mix(predict,predict1,predict2)
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
            (torch.mean(self.parameters0 + torch.mean(self.parameters1) + torch.mean(self.parameters2)))
        return x
##########################################################################################
class Classfier_cluter(nn.Module):
    def __init__(self,embedding_dimension,
        cluster_number,
        alpha=1.0,
        cluster_centers = None,):
        super(Classfier_cluter, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)

        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=False)
    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1.0) / 2.0
        numerator = numerator ** power
        predict = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return predict



# # 加载模型
# model = YourModel()
#
# # 获取状态字典
# state_dict = model.state_dict()
#
# # 修改参数
# new_param = torch.randn(4, 300)
# state_dict['classifier_c.weight'] = new_param
#
# # 将修改后的状态字典加载回模型中
# model.load_state_dict(state_dict)


# 这个运算是用来计算batch和self.cluster_centers之间的欧几里得距离的平方。
# 首先，我们使用torch.unsqueeze()方法将batch的维度扩展到(batch_size, 1, dim)，
# 然后使用广播机制将self.cluster_centers的维度扩展到(1, num_clusters, dim)。
# 接下来，我们计算两个张量之间的差值，即(batch_size, num_clusters, dim)。
# 然后，我们对差值的每个元素进行平方，即(batch_size, num_clusters, dim)。
# 最后，我们使用torch.sum()方法沿着最后一个维度求和，即(batch_size, num_clusters)，
# 得到每个样本与每个聚类中心之间的欧几里得距离的平方。这个运算通常用于聚类算法中。希望这可以帮到你！