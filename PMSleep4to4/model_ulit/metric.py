import torch
def hot_y(y,type_n):
    y_hot = torch.zeros(y.shape[0],type_n).to(y.device)
    place = torch.arange(y.shape[0],dtype=torch.long)
    y = y.to(torch.long)
    y_hot[place,y] = 1
    return y_hot

def Metric_c(output1, target1,device):
    type_n = output1.shape[1]
    target1 = hot_y(target1,type_n)

    pred1 = torch.argmax(output1, dim=1).to(device)
    target1 = torch.argmax(target1, dim=1).to(device)
    #获得pred2长度的全0数组
    assert pred1.shape[0] == len(target1)
    #计算准确率
    correct1 = 0
    correct1 += torch.sum(pred1 == target1).item()
    return correct1/len(target1)
def Metric_p(output1, target1):
    type_n = output1.shape[1]

    pred = torch.argmax(output1, dim=1)
    class_target = target1.shape[1]
    target = torch.argmax(target1, dim=1)
    assert pred.shape[0] == len(target)
    # 分类计算混淆矩阵
    TP = [0 for i in range(class_target)]
    FP = [0 for i in range(class_target)]
    TN = [0 for i in range(class_target)]
    FN = [0 for i in range(class_target)]
    class_acc = [0 for i in range(class_target)]
    class_precision = [0 for i in range(class_target)]
    class_recall = [0 for i in range(class_target)]
    class_F1 = [0 for i in range(class_target)]
    for i in range(class_target):
        TP[i] += torch.sum((pred == i) & (target == i)).item()
        FP[i] += torch.sum((pred == i) & (target != i)).item()
        TN[i] += torch.sum((pred != i) & (target != i)).item()
        FN[i] += torch.sum((pred != i) & (target == i)).item()
        # 计算准确率,精确率,召回率,F1，同时在计算时避免除0错误
        class_acc[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
        class_precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0
        class_recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0
        class_F1[i] = 2 * class_precision[i] * class_recall[i] / (
                    class_precision[i] + class_recall[i]) if (class_precision[i] + class_recall[i]) != 0 else 0
        #求均值

    F1 = sum(class_F1) / type_n
    return F1 , class_F1

def class_metric(output, target):
    pred = torch.argmax(output, dim=1)
    class_target = target.shape[1]
    target = torch.argmax(target, dim=1)
    assert pred.shape[0] == len(target)
    #分类计算混淆矩阵
    TP = [0 for i in range(class_target)]
    FP = [0 for i in range(class_target)]
    TN = [0 for i in range(class_target)]
    FN = [0 for i in range(class_target)]
    class_acc = [0 for i in range(class_target)]
    class_precision = [0 for i in range(class_target)]
    class_recall = [0 for i in range(class_target)]
    class_F1 = [0 for i in range(class_target)]
    class_distribution = torch.zeros((class_target, class_target+1))
    for i in range(class_target):
        TP[i] += torch.sum((pred == i) & (target == i)).item()
        FP[i] += torch.sum((pred == i) & (target != i)).item()
        TN[i] += torch.sum((pred != i) & (target != i)).item()
        FN[i] += torch.sum((pred != i) & (target == i)).item()
        #计算准确率,精确率,召回率,F1，同时在计算时避免除0错误
        class_acc[i] = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
        class_precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0
        class_recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0
        class_F1[i] = 2 * class_precision[i] * class_recall[i] / (class_precision[i] + class_recall[i]) if (class_precision[i] + class_recall[i]) != 0 else 0

        class_distribution[i][0] = torch.sum(target == i).item()
        if class_distribution[i][0] != 0:
            for j in range(class_target):
                class_distribution[i][j+1] = int(torch.sum((pred == j) & (target == i)).item())
    # 将class_distribution
    class_distribution_1 = class_distribution[:, 0]
    #将class_precision转换成tensor
    class_precision = torch.tensor(class_precision)
    class_recall = torch.tensor(class_recall)
    #计算M_F1
    M_precision = torch.sum(class_precision * class_distribution_1) / torch.sum(class_distribution_1)
    M_recall = torch.sum(class_recall * class_distribution_1) / torch.sum(class_distribution_1)
    M_F1 = 2 * M_precision * M_recall / (M_precision + M_recall)

    return class_acc, class_precision, class_recall, class_F1, class_distribution, M_F1.item()


