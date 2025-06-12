# 郑德钦  2024/3/1
import argparse
from function import Config, cost_time
from sleepdata.dataset import *
from model_ulit.metric import class_metric
from model_ulit.model import FeatureExtractor,Classifier
from function import cost_time
import pandas as pd
import torch
import torch.nn as nn
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
    output_list_y2_0 = []
    target_list_y2_0 = []
    output_list_y2_1 = []
    target_list_y2_1 = []
    Fmodel.eval()
    Cmodel.eval()
    with torch.no_grad():
        for i in range(len(data_x_v)):
            for j in range(0, len(data_x_v[i]), 128):
                if j + 128 < len(data_x_v[i]):
                    data, target, p1 = data_x_v[i][j:j + 128], data_y_v[i][j:j + 128], data_prompt_v[i][
                                                                                                  j:j + 128]
                    data_y2 = data_y2_v[i][j:j + 128].to(device)

                else:
                    data, target, p1 = data_x_v[i][-128:], data_y_v[i][j:].to(device), data_prompt_v[i][-128:]
                    data_y2 = data_y2_v[i][j:].to(device)

                x = Fmodel(data.to(device), p1.to(device))
                x = x.detach().unfold(dimension=0, size=config.config["data_set"]["sq_len"], step=1).transpose(1, 2)
                output = Cmodel(x.detach())
                output = torch.cat([output[0, 0, :].unsqueeze(0), output[:, 1, :], output[-1, 2, :].unsqueeze(0)],
                                   dim=0)
                if j + 128 > len(data_x_v[i]):
                    real_len = len(data_x_v[i]) - j
                    output = output[-real_len:]
                place_y2_0 = torch.where(data_y2 == 0)[0]
                place_y2_1 = torch.where(data_y2 == 1)[0]
                output1_0 = output[place_y2_0]
                output1_1 = output[place_y2_1]
                target1_0 = output[place_y2_0]
                target1_1 = output[place_y2_1]
                if len(place_y2_0) != 0:
                    if output_list_y2_0 == []:
                        output_list_y2_0 = output1_0
                        target_list_y2_0 = target1_0
                    else:
                        output_list_y2_0 = torch.cat((output_list_y2_0, output1_0), dim=0)
                        target_list_y2_0 = torch.cat((target_list_y2_0, target1_0), dim=0)
                if len(place_y2_1) != 0:
                    if output_list_y2_1 == []:
                        output_list_y2_1 = output1_1
                        target_list_y2_1 = target1_1
                    else:
                        output_list_y2_1 = torch.cat((output_list_y2_1, output1_1), dim=0)
                        target_list_y2_1 = torch.cat((target_list_y2_1, target1_1), dim=0)
    class_acc, class_precision, class_recall, class_F1, class_distribution, Mf1 = class_metric(output_list_y2_0, target_list_y2_0)
    class_acc1, class_precision1, class_recall1, class_F11, class_distribution1, Mf11 = class_metric(output_list_y2_1, target_list_y2_1)
    #写入excel文件
    torch.set_printoptions(precision=4, sci_mode=False)
    with open(config.config["trainer"]["save_dir"] + '/classfier_result0/'+str(
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
    with open(config.config["trainer"]["save_dir"] + '/classfier_result1/'+str(
                               config.config["data_set"]["sleepedf_split"][1])+'.txt', 'w') as f:
        f.write('对于正常信息，78个数据中的测试，表现如下 ' + '\n')
        #保留四位小数写入
        f.write('准确率：' + str(class_acc1) + '\n')
        f.write('精确率：' + str(class_precision1) + '\n')
        f.write('召回率：' + str(class_recall1) + '\n')
        f.write('F1值：' + str(class_F11) + '\n')
        #混淆矩阵以整数写入,不使用科学计数法
        f.write('混淆矩阵：' + '\n')
        f.write(str(class_distribution1) + '\n')
        #写入Mf1
        f.write('Mf1：' + str(Mf11) + '\n')





if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-da', '--np_data_dir', default=r"E:\sleepstage\sleepdata\2013p", type=str,
                      help='Directory containing numpy files')

    options = []
    args2 = args.parse_args()
    config = Config(args2.config)
    # 检查config中的save_dir是否存在，不存在就创建
    if not os.path.exists(config.config["trainer"]["save_dir"]):
        os.makedirs(config.config["trainer"]["save_dir"])
    fold_all = int(1 / (1 - config.config["data_set"]["sleepedf_split"][0]))
    if config.config["best_f1"] == []:
        # 创建fold_all个列表，用于存放每个fold的最佳f1
        config.config["best_f1"] = [0 for i in range(fold_all)]
    if config.config["best_f1_w"] == []:
        # 创建fold_all个列表，用于存放每个fold的最佳f1
        config.config["best_f1_w"] = [0 for i in range(fold_all)]
    if config.config["cost_time"] == []:
        config.config["cost_time"] = [0 for i in range(fold_all)]
    while True:
        for fold in range(1, fold_all + 1):
            config.config["data_set"]["sleepedf_split"][1] = fold
            main_evaluate(config, args2)


