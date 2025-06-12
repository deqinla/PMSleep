import matplotlib.pyplot as plt
from sleepdata.dataset import *
from model_ulit.metric import class_metric
from model_ulit.model import FeatureExtractor,Classifier
from function import cost_time
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
@cost_time
def main_evaluate(config, args):
    batch_size = config.config["data_loader"]["args"]["batch_size"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #读取测试集
    data_x_v, data_y_v, data_prompt_v, data_y2_v, channels_v, freq_v, type_n_v \
        = val_data_load(args.np_data_dir, config.config["data_set"]["sleepedf_split"]
                        )
    #读取模型
    Fmodel = FeatureExtractor(3000, freq_v,
                              len(channels_v), config.config["model"], type_n_v).to(device)
    Cmodel = Classifier(config.config["model"], type_n_v).to(device)
    Fmodel.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
                               config.config["data_set"]["sleepedf_split"][1]) + "best_model_f1_all_F.pth"))
    Cmodel.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_model_f1_all_C.pth"))
    output_list = []
    target_list = []
    with torch.no_grad():
        Fmodel.eval()
        Cmodel.eval()
        for i in range(len(data_x_v)):
            for j in range(0,len(data_x_v[i]), 128):
                if j + 128 < len(data_x_v[i]):
                    data, target,p1 = data_x_v[i][j:j + 128], data_y_v[i][j:j + 128].to(device),data_prompt_v[i][j:j + 128]

                else:
                    data, target,p1 = data_x_v[i][-128:], data_y_v[i][j:].to(device),data_prompt_v[i][-128:]
                x = Fmodel(data.to(device), p1.to(device))
                x = rearrange(x, '(b l)  n -> b l n',l = config.config["data_set"]["sq_len"]).to(device).detach()
                output = Cmodel(x.detach())
                output = rearrange(output, 'b l n -> (b l)  n',l = config.config["data_set"]["sq_len"])
                if j + 128 > len(data_x_v[i]):
                    real_len = len(data_x_v[i]) - j
                    output  = output[-real_len:]
                if output_list == []:
                    output_list = output
                    target_list = target
                else:
                    output_list = torch.cat((output_list, output), dim=0)
                    target_list = torch.cat((target_list, target), dim=0)
    class_acc, class_precision, class_recall, class_F1, class_distribution, Mf1 = class_metric(output_list, target_list)
    #写入excel文件
    torch.set_printoptions(precision=4, sci_mode=False)
    with open(config.config["trainer"]["save_dir"] + '/classfier_result/'+str(
                               config.config["data_set"]["sleepedf_split"][1])+'.txt', 'w') as f:
        f.write('对于正常信息，78个数据中的测试，表现如下 ' + '\n')
        #保留四位小数写入
        f.write('准确率：' + str(class_acc) + '\n')
        f.write('精确率：' + str(class_precision) + '\n')
        f.write('召回率：' + str(class_recall) + '\n')
        f.write('F1值：' + str(class_F1) + '\n')
        #混淆矩阵以整数写入,不使用科学计数法
        f.write('混淆矩阵：' + '\n')
        f.write(str(class_distribution) + '\n')
        #写入Mf1
        f.write('Mf1：' + str(Mf1) + '\n')


