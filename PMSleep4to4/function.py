import os
import json
import time
import torch
import numpy as np
from openpyxl import Workbook
import re
import pandas as pd
import copy
class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.config = json.load(f)

    def update(self, key, value):
        self.config[key] = value
        with open(self.file_path, 'w') as f:
            json.dump(self.config, f, indent=4)



def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def txt_to_excel(path):
    # 从txt文件中读取数据
    with open(path, 'r') as file:
        content = file.read()

    # 从文本中提取混淆矩阵,以混淆矩阵字样开始，最后一行是MF1，不属于混淆矩阵
    match = re.search(r'混淆矩阵：\n(.*?)\nMf1', content, re.S)
    confusion_matrix_str = match.group(1).strip()

    # 去除字符串中的 "tensor([[" 和 "]]"
    confusion_matrix_str = confusion_matrix_str.replace("tensor([[", "").replace("]])", "")
    confusion_matrix_str = confusion_matrix_str.replace("]", "").replace("[", "")
    confusion_matrix_str = confusion_matrix_str.replace(".", "")
    confusion_matrix_str = confusion_matrix_str.replace(" ", "")
    # 转换为numpy数组
    confusion_matrix = np.array([list(map(int, confusion_matrix_str.split(',')))])
    confusion_matrix = confusion_matrix.reshape(5,6)

    return confusion_matrix

if __name__ == '__main__':
    path = r".\saved\classfier_result\1.txt"
    c = None
    for i in range(1,11):
        txt_name = str(i) + ".txt"
        path_i = path.replace('1.txt', txt_name)
        ci = txt_to_excel(path_i)
        if c is None:
            c = ci
        else:
            c = c + ci
    #计算recall,precision,f1
    recall = []
    precision = []
    f1 = []
    for i in range(5):
        recall.append(c[i][i+1] / c[i][0])
        precision.append(c[i][i+1] / np.sum(c[:, i+1]))
        f1.append(2 * recall[i] * precision[i] / (recall[i] + precision[i]))
    #计算mF1
    M_acc = np.sum(np.diag(c[:,1:])) / np.sum(c[:,1:])
    M_P = sum(precision * c[:, 0]) / np.sum(c[:, 0])
    M_recall = sum(precision * c[:, 0]) / np.sum(c[:, 0])
    M_F1 = 2 * M_P * M_recall / (M_P + M_recall)
    #计算kappa系数
    p0 = np.sum(np.diag(c[:,1:])) / np.sum(c[:,1:])
    pc = 0
    for i in range(5):
        fenzi = np.sum(c[i, 1:])/np.sum(c[:, 0])
        fenmu = np.sum(c[:, i+1])/np.sum(c[:, 0])
        pc += fenzi * fenmu
    kappa = (p0 - pc) / (1 - pc)

    #把c转换成list
    c = c.tolist()
    #将numpy的c,reacall,precision,f1写入 excel
    wb = Workbook()
    ws = wb.active
    ws.append(['混淆矩阵'])
    for i in range(5):
        ws.append(c[i])
    ws.append(['recall'])
    ws.append(recall)
    ws.append(['precision'])
    ws.append(precision)
    ws.append(['f1'])
    ws.append(f1)
    #写入M_F1,M_acc,M_P,M_recall,M_F1，四个写成一行
    ws.append([ 'M_acc','M_F1', 'M_P', 'M_recall', 'kappa'])
    ws.append([M_acc, M_F1, M_P, M_recall, kappa])
    wb.save(path.replace('1.txt', 'result_2.4.1.xlsx'))