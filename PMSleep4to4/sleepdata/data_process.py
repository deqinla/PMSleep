import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
over_type = 1

def process_data(psg_fnames,save_path):
    # 读取原始数据，如果没有保存过，则保存
    data_x, data_y, data_prompt, data_y2, channels, freq, type_n = read_data_flod(psg_fnames)
    # 保存x=data_x, y=data_y, channels=channels,data_len=data_len, freq=freq, type_n=type_n为.pt文件
    data = {"x": data_x, "y": data_y, "x_prompt": data_prompt, "y2": data_y2,
            "channels": channels, "freq": freq,
            "type_n": type_n}
    torch.save(data, save_path)

def read_data_flod(psg_fnames):
    # 查找有几个文件
    data_x = []
    data_prompt = []
    data_y = []
    data_y2 = []


    for file in range(len(psg_fnames)):
        data = np.load(psg_fnames[file])
        if file == 0:
            data_x = [torch.tensor(data["x"], dtype=torch.float32)]
            data_prompt = [torch.tensor(data["x_prompt"], dtype=torch.float32)]
            dy = torch.tensor(data["y"], dtype=torch.float32)
            type_y = torch.unique(dy)
            type_n = type_y.shape[0]
            data_y.append(hot_y(dy, 5))

            data_y2.append(change_labe(dy))
        else:
            data_x.append(torch.tensor(data["x"], dtype=torch.float32))
            data_prompt.append(torch.tensor(data["x_prompt"], dtype=torch.float32))
            dy = torch.tensor(data["y"], dtype=torch.float32)
            data_y.append(hot_y(dy, 5))

            data_y2.append(change_labe(dy))

    channels = data["ch_label"]
    freq = data["fs"]
    return data_x, data_y,data_prompt,data_y2, channels, freq, type_n



def over_samping(data_x, data_y, data_prompt,data_y2):
    data_len = data_x.shape[2]
    data_x_expand = []
    data_y_expand = []
    data_prompt_expand = []
    data_y2_expand = []

    place = torch.where(data_y[:,over_type] == 1)[0]
    for k in range(1, len(place)):
        if place[k] - place[k - 1] == 1:
            data_x_new = torch.cat((data_x[place[k - 1], :, data_len // 2:],
                                    data_x[place[k], :, 0:data_len // 2]), dim=1).unsqueeze(0)
            data_prompt_new = torch.cat((data_prompt[place[k - 1], :, data_len // 2:],
                                         data_prompt[place[k], :, 0:data_len // 2]), dim=1).unsqueeze(0)
            data_y_new = data_y[place[k]].unsqueeze(0)
            data_y2_new = data_y2[place[k]].unsqueeze(0)

            data_x_expand.append(data_x_new)
            data_y_expand.append(data_y_new)
            data_prompt_expand.append(data_prompt_new)

            data_y2_expand.append(data_y2_new)
    return torch.vstack(data_x_expand), torch.vstack(data_y_expand), torch.vstack(data_prompt_expand),\
        torch.vstack(data_y2_expand).squeeze(1)


def generate_prompts(input_tensor, pad_size):
    # 确定填充尺寸，左侧填充3个元素
    # 在最后一个维度左侧填充pad_size个0
    padded = F.pad(input_tensor, (pad_size, 0))
    # 使用unfold创建滑动窗口（窗口大小3，步长1）
    unfolded = padded.unfold(-1, pad_size, 1)
    # 截取前N个窗口（N为原始序列长度）
    prompts = unfolded[..., :input_tensor.size(-1), :]
    return prompts

#生成下一个阶段是否有变换的标签
def change_labe(input_tensor):
    later1 = F.pad(input_tensor, (0, 1))[..., 1:]
    data_y2 = input_tensor != later1
    data_y2 = torch.tensor(data_y2, dtype=torch.int)
    return data_y2

def hot_y(y, type_n):
    y_hot = torch.zeros(y.shape[0], type_n).to(y.device)
    place = torch.arange(y.shape[0], dtype=torch.long)
    y = y.to(torch.long)
    y_hot[place, y] = 1
    return y_hot