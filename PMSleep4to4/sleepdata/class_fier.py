import glob
import numpy as np
import os
import shutil
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

dataset_path = 'sleepedf/sleep-cassette/'
saved_path = 'sleepedf/sleep-cassette_classfied/'
save_list = [0,1,2,3,4,5]
# 保存地址，如果不存在就创建，如果存在就删除重建
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
else:
    shutil.rmtree(saved_path)
    os.makedirs(saved_path)

# 读取数据
npz_fnames_all = glob.glob(os.path.join(dataset_path, "*.npz") )
#排序
npz_fnames_all.sort()
data_classfied = []
    # 读取第一个文件,0.8代表80%的数据用于训练，前20%的数据用于测试，这里只分类测试数据
data_classfied = []
npz_fnames = npz_fnames_all[0:int(0.8*len(npz_fnames_all))]
npz_file = npz_fnames[0]
data = np.load(npz_file)
data_x = data["x"]
data_y = data["y"]
channel_names = data["channel_names"]
fres = data["fs"]

for annindex in save_list:
    data_place = np.where(data_y == annindex)
    data_place = data_place[0]
    # 保存数据
    clssfiy_data = data_x[data_place]
    if len(data_classfied) == 0:
        data_classfied = [clssfiy_data]
    else:
        data_classfied.append(clssfiy_data)
for npz_index in range(0,len(npz_fnames)):
    npz_file = npz_fnames[npz_index]
    data = np.load(npz_file)
    data_x = data["x"]
    data_y = data["y"]
    channel_names = data["channel_names"]
    fres = data["fs"]

    # 将数据分类
    for annindex in save_list:
        data_place = np.where(data_y == annindex)
        data_place = data_place[0]
        # 保存数据
        clssfiy_data = data_x[data_place]
        if len(data_classfied[annindex]) == 0:
            data_classfied[annindex] = clssfiy_data
        else:
            data_classfied[annindex] = np.concatenate((data_classfied[annindex], clssfiy_data), axis=0)
#比较数据，找到数据最少的类别
min_len = data_classfied[0].shape[0]
for index_min in range(1,len(data_classfied)):
    if min_len > data_classfied[index_min].shape[0] and data_classfied[index_min].shape[0] != 0:
        min_len = data_classfied[index_min].shape[0]
print("min",min_len)
# 保存数据
for saveindex in range(len(save_list)):
    # 保存数据
    # 设置保存路径
    save_dir = saved_path + "\\" +str(saveindex) + ".npz"
    np.savez(save_dir, x=data_classfied[saveindex], y=save_list[saveindex],
             channel_names=channel_names, fs=fres,n_samples=data_classfied[saveindex].shape[2],n_classfier=data_classfied[saveindex].shape[0]
             , n_min= min_len)
    print(data_classfied[saveindex].shape[0])




