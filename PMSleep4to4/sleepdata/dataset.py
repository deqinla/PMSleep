import torch
import torch.nn as nn
import os
from sleepdata.data_process import process_data
import numpy as np
import glob

class sleppEDF_train(nn.Module):
    def __init__(self, root_path_normal,  split,sq_len,sq_step):
        # 记录有几种通道组合
        super(sleppEDF_train).__init__()
        self.root_path = root_path_normal
        self.split = split
        self.sq_len = sq_len
        self.sq_step = sq_step
        self.fold_num = int(1 / (1 - self.split[0]))
        (self.data_x, self.data_y, self.data_prompt,self.data_y2,self.channels,
         self.data_len_list,self.data_len, self.freq, self.type_n) = self.__read_data_train__()


    def __read_data_train__(self):
        psg_fnames = glob.glob(os.path.join(self.root_path, "*.npz"))
        psg_fnames.sort()
        # 将psg_fnames拆分成self.fold_num份
        psg_fnames = np.array(psg_fnames)
        psg_fnames = np.array_split(psg_fnames, self.fold_num)
        psg_fnames = [list(arr) for arr in psg_fnames]
        save_path_f = os.path.join(self.root_path, str("readed_s"))
        if not os.path.exists(save_path_f):
            # 创建文件夹
            os.makedirs(save_path_f)
        # 只取验证集的编号self.split[1]
        val_pick_num = self.split[1] - 1
        # 根据self.flag判断是训练集还是验证集,并返回对应的数据
        data_x = []
        data_y = []
        data_prompt = []
        data_y2 = []
        for i in range(len(psg_fnames)):
            if i == val_pick_num:
                continue
            save_name = self.root_path.split("\\")[-1] + '_' +str(i) +".pt"
            save_path = os.path.join(save_path_f, save_name)
            # 读取原始数据，如果没有保存过，则保存
            if not os.path.exists(save_path):
                process_data(psg_fnames[i],save_path)
            data = torch.load(save_path,weights_only=False)
            data_xi, data_yi, channels, freq, type_n, data_prompti,  data_y2i = data["x"], data["y"], data["channels"], \
                data["freq"], data["type_n"],data["x_prompt"],data["y2"]
            data_x += data_xi
            data_y += data_yi
            data_prompt += data_prompti
            data_y2 += data_y2i

        data_len_list = [0]
        for i in range(len(data_x)):
            data_len_list.append(data_len_list[-1] + data_x[i].shape[0])

        data_len = data_len_list[-1]

        return data_x, data_y, data_prompt,data_y2,channels, data_len_list,data_len, freq, type_n

    def __getitem__(self, index):
        index = index * self.sq_step
        index = index + torch.randint(0, self.sq_step,  (1,))
        for i in range(1,len(self.data_len_list)):
            if index >= self.data_len_list[i]:
                continue
            reindex = index - self.data_len_list[i]
            pindex = index - self.data_len_list[i-1]
            if pindex < self.sq_len:
                return self.data_x[i-1][:self.sq_len], self.data_y[i-1][:self.sq_len], self.data_prompt[i-1][:self.sq_len],self.data_y2[i-1][:self.sq_len]
            elif reindex > -self.sq_len:
                return self.data_x[i-1][-self.sq_len:], self.data_y[i-1][-self.sq_len:], self.data_prompt[i-1][-self.sq_len:],self.data_y2[i-1][-self.sq_len:]
            else:
                return (self.data_x[i-1][pindex:pindex+self.sq_len], self.data_y[i-1][pindex:pindex+self.sq_len],
                        self.data_prompt[i-1][pindex:pindex+self.sq_len],self.data_y2[i-1][pindex:pindex+self.sq_len])
    def __len__(self):
        return self.data_len//self.sq_step

def val_data_load(root_path_normal, split):
    fold_num = int(1 / (1 - split[0]))
    psg_fnames = glob.glob(os.path.join(root_path_normal, "*.npz"))
    psg_fnames.sort()
    # 将psg_fnames拆分成self.fold_num份
    psg_fnames = np.array(psg_fnames)
    psg_fnames = np.array_split(psg_fnames, fold_num)
    psg_fnames = [list(arr) for arr in psg_fnames]
    save_path_f = os.path.join(root_path_normal, str("readed_s"))
    if not os.path.exists(save_path_f):
        # 创建文件夹
        os.makedirs(save_path_f)
    # 只取验证集的编号self.split[1]
    val_pick_num = split[1] - 1
    save_name = root_path_normal.split("\\")[-1] + '_' + str(val_pick_num) + ".pt"
    save_path = os.path.join(save_path_f, save_name)
    if not os.path.exists(save_path):
        process_data(psg_fnames[val_pick_num], save_path)
    data = torch.load(save_path, weights_only=False)
    data_x, data_y, channels, freq, type_n, data_prompt, data_y2 = data["x"], data["y"], data["channels"], \
        data["freq"], data["type_n"], data["x_prompt"], data["y2"]


    return data_x, data_y, data_prompt,data_y2,channels, freq, type_n


