import os
import numpy as np

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

import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET

###############################
EPOCH_SEC_SIZE = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"E:\sleepstage\sleepdata\shhs\polysomnography\edfs\shhs1",
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default=r"E:\sleepstage\sleepdata\shhs\polysomnography\annotations-events-profusion\shhs1",
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default=r"./shhsnpz",
                        help="Directory where to save numpy files outputs.")

    args = parser.parse_args()
    # 选择的通道，例如EEG(sec), ECG, EMG, EOG(L), EOG(R), EEG
    select_ch = ["ECG", "EMG", "EOG(L)", "EOG(R)"]
    #时间采样频率
    prompt_rate = 1.0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    edf_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.ann_dir, "*-profusion.xml"))
    edf_fnames.sort()
    ann_fnames.sort()
    for file_id in range(0,len(edf_fnames)):
        savename = os.path.basename(edf_fnames[file_id]).replace(".edf",  ".npz")
        savename =  os.path.join(args.output_dir, savename)
        if os.path.exists(savename):
            print("Already exists: ", savename)
            continue
        if os.path.exists(os.path.join(args.output_dir, edf_fnames[file_id].split('/')[-1])[:-4]+".npz"):
            continue
        print(edf_fnames[file_id])

        raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()




        columns_all = raw_ch_df.columns
        #选取columns_all中包含EEG字段的要素
        for cc in columns_all:
            if 'EEG'in cc: select_ch.append(cc)
        raw_ch_data = raw_ch_df[select_ch].values
        if sampling_rate == 250:
            #降采样到125
            raw_ch_data = raw_ch_data[::2,:]
            sampling_rate_now = sampling_rate // 2
        else:
            sampling_rate_now = sampling_rate


        #剩下作为事件数据
        prompt_channls =['SaO2', 'POSITION',  'ABDO RES', 'H.R.', 'THOR RES']
        prompt_data = raw_ch_df[prompt_channls].values



    ###################################################
        labels = []
        # Read annotation and its header
        t = ET.parse(ann_fnames[file_id])
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            # if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
            #     faulty_File = 1

        # if faulty_File == 1:
        #     print( "============================== Faulty file ==================")
        #     continue

        labels = np.asarray(labels)

        # Remove movement and unknown stages if any
        print(raw_ch_data.shape)

        # Verify that we can split into 30-s epochs
        if len(raw_ch_data) % (EPOCH_SEC_SIZE * sampling_rate_now) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch_data) / (EPOCH_SEC_SIZE * sampling_rate_now)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch_data, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)
        prompt_data = np.asarray(np.split(prompt_data, n_epochs)).astype(np.float32)
        #将prompt_data沿着第二个维度降采样到真实的频率,例如1hz就sampling_rate范围取第一个就行
        prompt_data = prompt_data[:, ::int(sampling_rate / prompt_rate),:]


        print(x.shape)
        print(y.shape)
        print(prompt_data.shape)
        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        prompt_data = prompt_data[select_idx]

        #去除unknown的数据
        place = np.where(y > 5)[0]
        if len(place) > 0:
            x = np.delete(x, place, axis=0)
            y = np.delete(y, place, axis=0)
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Saving as numpy files

        save_dict = {
            "x": x,
            "y": y,
            "x_prompt": prompt_data,
            "fs": sampling_rate_now,
            "ch_label": select_ch,
        }
        np.savez(savename, **save_dict)
        print(" ---------- Done this file ---------")
    print("-----------finished---------")


if __name__ == "__main__":
    main()
