# 郑德钦  2024/3/1
import argparse
import json
import time
import numpy as np
from sleepdata.dataset import *
from function import Config, cost_time
from train import main_train
from evaluate import main_evaluate
import torch

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-da', '--np_data_dir', default=r"E:\hit\graduate\sleepdata\2013p", type=str,
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
    for fold in range(1, fold_all + 1):
        config.config["data_set"]["sleepedf_split"][1] = fold
        print("开始计算第", fold, "折实验,该折最佳f1目前为", config.config["best_f1"][fold - 1])

        # 记录开始时间
        start_time = time.time()

        config.config["best_f1"][fold - 1] = main_train(config, args2, config.config["best_f1"][fold - 1])
        print("目前f1最佳", config.config["best_f1"][fold - 1])
        config.config["best_f1_w"][fold - 1] = config.config["optimizer"]["args"]["weight_decay"]

        # 记录结束时间
        end_time = time.time()
        # 记录时间
        print("第", fold, "折实验耗时", end_time - start_time, "s")
        config.config["cost_time"][fold - 1] = end_time - start_time

        main_evaluate(config, args2)
        # 写入config
        with open(args2.config, "w") as f:
            json.dump(config.config, f, indent=4)

