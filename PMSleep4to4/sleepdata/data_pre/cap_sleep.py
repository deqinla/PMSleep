import os
import signal
import chardet
import numpy as np
import time
import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2

from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from mne.io import concatenate_raws, read_raw_edf
from scipy.signal import resample as scipy_resample

# 建立一个字典，用于存储不同类型的文件名称关键词
# n1-n16无病理学（对照）
# brux 1-brux 2 Bruxism
# ins 1-ins 9失眠症
# narco 1-narco 5嗜睡症SY
# nfle 1-nfle 40夜间额叶癫痫
# plm 1-plm 10周期性腿部运动
# rbd 1-srbd 22 REM行为障碍
# sdb 1-sdb 4睡眠呼吸障碍
ann_dict = {
    0: ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16'],
    1: ['brux1', 'brux2'],
    2: ['ins1', 'ins2', 'ins3', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9'],
    3: ['narco1', 'narco2', 'narco3', 'narco4', 'narco5'],
    4: ['nfle1', 'nfle2', 'nfle3', 'nfle4', 'nfle5', 'nfle6', 'nfle7', 'nfle8', 'nfle9', 'nfle10', 'nfle11', 'nfle12',
        'nfle13', 'nfle14', 'nfle15', 'nfle16', 'nfle17', 'nfle18', 'nfle19', 'nfle20', 'nfle21', 'nfle22', 'nfle23',
        'nfle24', 'nfle25', 'nfle26', 'nfle27', 'nfle28', 'nfle29', 'nfle30', 'nfle31', 'nfle32', 'nfle33', 'nfle34',
        'nfle35', 'nfle36', 'nfle37', 'nfle38', 'nfle39', 'nfle40'],
    5: ['plm1', 'plm2', 'plm3', 'plm4', 'plm5', 'plm6', 'plm7', 'plm8', 'plm9', 'plm10'],
    6: ['rbd1', 'rbd2', 'rbd3', 'rbd4', 'rbd5', 'rbd6', 'rbd7', 'rbd8', 'rbd9', 'rbd10', 'rbd11', 'rbd12', 'rbd13',
        'rbd14',
        'rbd15', 'rbd16', 'rbd17', 'rbd18', 'rbd19', 'rbd20', 'rbd21', 'rbd22'],
    7: ['sdb1', 'sdb2', 'sdb3', 'sdb4']
}

# 建立一个字典，用于存储不同类型的事件
ABevent_dict = {
    "MCAP-A1": 1,
    "MCAP-A2": 2,
    "MCAP-A3": 3,
}
# W=清醒，S1-S4=睡眠阶段，R=REM，MT=身体运动
stage_dict = {
    "W": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 3,
    "REM": 4,
    "R": 4,
    "MT": 5,
}
traget_channel = ["C4-A1", "C4-P4", "F4-C4"]
###############################
EPOCH_SEC_SIZE = 30


def resample(raw_data, freq, target_freq):
    if freq == target_freq:
        return raw_data
    elif freq == 512:
        # 512 Hz -> 256 Hz,取偶数点
        raw_data = raw_data.T
        raw_data = raw_data[::2, :]
        return raw_data.T
    else:
        # 使用men库将数据重采样为256Hz
        raw_data = raw_data.T
        raw_data = scipy_resample(raw_data, int(raw_data.shape[0] / freq * target_freq), axis=0)
        return raw_data.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="E:\sleepstage\sleepdata\cap-sleep-database-1.0.0",
                        help="File path to the PSG files.")
    parser.add_argument("--output_dir", type=str, default="E:\sleepstage\sleepdata\cap_npz",
                        help="Directory where to save numpy files outputs.")
    args = parser.parse_args()
    # 分类获得edf文件和label文件
    edf_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    label_fnames = glob.glob(os.path.join(args.data_dir, "*.txt"))
    edf_fnames.sort()
    label_fnames.sort()
    # 按照edf文件的名称，将错误的txt文件从label_fnames中删除
    for i in range(len(edf_fnames)):
        while edf_fnames[i][:-4] != label_fnames[i][:-4]:
            del label_fnames[i]

    # 按照dict给edf文件分类，先给两个文件列表添加一个维度
    len_dict = len(ann_dict)
    file_list = []
    for i in range(len_dict):
        # 为dict中的每个key创建一个文件夹
        filename = ann_dict[i][0][:-1]
        file_list.append(os.path.join(args.output_dir, filename))
        if not os.path.exists(file_list[i]):
            os.makedirs(file_list[i])

    # 遍历edf文件
    for i in range(0, len(edf_fnames)):
        raw = read_raw_edf(edf_fnames[i])
        # 获取采样频率
        freq = raw.info['sfreq']
        # 如果采样频率不是256Hz，512，128就跳过
        if freq not in [256, 512, 128]:
            continue
        # 检查target_channel是否在通道中，不在继续，在就获取通道的位置
        traget_place = []
        for target in traget_channel:
            if target not in raw.ch_names:
                continue
            else:
                traget_place.append(raw.ch_names.index(target))
        if len(traget_place) != 3:
            continue

        # 获取数据
        raw_data = raw.get_data()
        # 转置数据，并将数据用重采样成采样频率=256
        raw_data = raw_data[traget_place, :]
        if freq != 256:
            raw_data = resample(raw_data, freq, 256)
            freq = 256

        # 获取开始时间
        start_time = raw.info['meas_date']
        start_time = start_time.strftime("%H:%M:%S")
        # 将时间转换为秒数
        start_time = [int(x) for x in start_time.split(":")]
        start_time = int(start_time[0] * 3600 + start_time[1] * 60 + start_time[2])

        # 计算数据长度
        data_len = raw_data.shape[1]

        # 检测文件编码
        with open(label_fnames[i], 'rb') as file:
            result = chardet.detect(file.read())
        # 使用检测到的编码打开文件
        # 读取label文件
        with open(label_fnames[i], 'r', encoding=result['encoding']) as file:
            label_data = file.read().splitlines()

        stage_event = np.ones(data_len)*-1
        # 遍历label文件，以记录事件
        for line in range(len(label_data)):
            # 寻找Sleep Stage	Time [hh:mm:ss]	Event	Duration[s]	Location所在行
            if label_data[line].find('Sleep Stage') == -1:
                continue
            else:
                break
        label_data = label_data[line + 1:]

        # 以空格分割每一行字符串
        label_data = [x.split("\t") for x in label_data]
        # 去除每一行中每个元素的空格,\t,unknown
        label_data = [[x for x in y if x != ''] for y in label_data]
        # 检查是否存在unknown的一列（不区分大小写）
        if 'N/A' in label_data[0][1] or 'unknown' in label_data[0][1].lower() or 'Prone' in label_data[0][1]:
            # 如果存在，删除unknown一列
            label_data = [x[:1] + x[2:] for x in label_data]
        flag = 0
        for line in range(len(label_data)):
            #检查持续时间
            line_duration = int((label_data[line][3]))
            if line_duration != 30:
                continue
            # 按照stage_dict将睡眠阶段转换为数字
            stage_event_line = (label_data[line][0])
            stage_event_line = stage_dict[stage_event_line]
            if stage_event_line == 0 and flag == 0:
                flag = 1
                continue
            # 获取事件开始时间
            line_time = (label_data[line][1])

            # 将事件开始时间与数据开始时间相减，得到事件开始时间点,分为夜晚和凌晨即24小时前和24小时后两种情况

            # 如果是.划分就按照.划分，如果是:划分就按照:划分
            if line_time.find('.') != -1:
                line_time = line_time.split(".")
            else:
                line_time = line_time.split(":")
            line_time = int(line_time[0]) * 3600 + int(line_time[1]) * 60 + int(line_time[2])
            if line_time < start_time :
                line_time += 24 * 3600
            # 计算差值
            line_time = int(line_time - start_time)
            # 将事件开始时间点转换为数据中的时间点
            line_time = int(line_time * freq)
            # 获取事件持续时间
            len_duration = int(line_duration * freq)
            if line_time > data_len:
                break
            # 将事件标记到数据中
            if line_time + len_duration > data_len:
                break
            stage_event[line_time:line_time + len_duration] = stage_event_line

        # 按照dict给edf文件分类，先给两个文件列表添加一个维度
        edf_name = edf_fnames[i].split('\\')[-1]
        edf_name = edf_name.split('.')[0]
        for ann_index in range(len_dict):
            if edf_name in ann_dict[ann_index]:
                filename = os.path.join(file_list[ann_index], edf_name)
                # 添加后缀
                filename = filename + '.npz'
                # 按照ch_place的信息，获取相同通道的数据
                break
        #去除标签是-1的数据
        sleep_idx = np.where(stage_event != -1)[0]
        raw_data = raw_data[:, sleep_idx]
        stage_event = stage_event[sleep_idx]

        # 去除睡前和睡后的数据
        sleep_idx = np.where(stage_event != 0)[0]
        raw_data = raw_data[:, sleep_idx[0]:sleep_idx[-1] + 1]
        stage_event = stage_event[sleep_idx[0]:sleep_idx[-1] + 1]

        epoch_len = int(30 * freq)
        epoch_num = int(raw_data.shape[1] / epoch_len)
        #如果数据长度不能整除30s，就去掉最后一段数据
        if raw_data.shape[1] % epoch_len != 0:
            raw_data = raw_data[:, :epoch_num * epoch_len]
            stage_event = stage_event[:epoch_num * epoch_len]
            epoch_num = int(raw_data.shape[1] / epoch_len)
        raw_data = np.reshape(raw_data, (raw_data.shape[0], epoch_num, epoch_len))
        stage_event = np.reshape(stage_event, (epoch_num, epoch_len))
        #stage只保留一位
        stage_event = stage_event[:, 0]
        raw_data = np.transpose(raw_data, (1, 0, 2))

        # 将数据和标签保存为npy文件
        save_dict = {
            "x": raw_data,
            "y": stage_event,
            "ch_label": traget_channel,
            "fs": freq,
        }
        print("读取的文件数量了：" + str(i) + "个。" + "最新位置为：" + filename)
        np.savez(filename, **save_dict)


if __name__ == "__main__":
    main()
