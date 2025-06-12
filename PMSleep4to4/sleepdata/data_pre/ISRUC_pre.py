import os
import glob
import ntpath
import logging
import argparse
import scipy.io as sio
import numpy as np

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,  # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./edf",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./npz",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="[EEG Fpz-Cz,EEG Pz-Oz,EOG horizontal]",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    mat_path = args.data_dir
    mat_files = glob.glob(os.path.join(mat_path, "*.mat"))

    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 创建日志文件
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # 读取数据
    for mat_file in mat_files:
        mat = sio.loadmat(mat_file)

if __name__ == "__main__":
    main()