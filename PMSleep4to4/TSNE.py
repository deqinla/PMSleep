from sklearn import manifold, datasets
from matplotlib.lines import Line2D  # 添加此导入语句
import matplotlib.pyplot as plt
from sleepdata.dataset import *
from model_ulit.model import FeatureExtractor,Classifier
from function import cost_time
import torch
from function import Config
import argparse
@cost_time
def main_evaluate(config, args):
    batch_size = config.config["data_loader"]["args"]["batch_size"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #读取测试集
    data_x_v, data_y_v, data_prompt_v, data_y2_v, channels_v, freq_v, type_n_v \
        = val_data_load(args.np_data_dir, config.config["data_set"]["sleepedf_split"]
                        )
    model = FeatureExtractor(3000, freq_v,
                              len(channels_v), config.config["model"], type_n_v).to(device)
    model.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
                               config.config["data_set"]["sleepedf_split"][1]) + "best_model_f1_all_F.pth"))
    model = model.to(device)
    x_in = data_x_v[0][:512]
    y_in = data_y_v[0][:512]

    model.eval()
    with torch.no_grad():
        save_name = str(fold) + "model.png"
        x = model(x_in.to(device), data_prompt_v[0][:512].to(device))
        TSNE(x.cpu().numpy(),y_in.cpu().numpy(),save_name)
        save_name = str(fold) + "origin.png"
        TSNE(x_in.cpu().numpy(),y_in.cpu().numpy(),save_name)

def TSNE(X, y,save_name):
    if len(X.shape) >= 3:
        X = X.reshape(X.shape[0], -1)
    y = y.argmax(1)  # 转换为类别标签

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(
        X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    plt.figure(figsize=(8, 8))

    # 新增：获取唯一类别标签
    unique_labels = np.unique(y)

    # 新增：创建图例代理对象
    proxies = []
    for label in unique_labels:
        proxies.append(Line2D([0], [0],
                              marker='o',
                              color=plt.cm.Set1(label / (len(unique_labels) - 1)),
                              linestyle='',
                              markersize=8))

    # 替换：用散点图代替文本标注
    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1],
                          c=y,  # 根据类别着色
                          cmap='Set1',  # 使用Set1颜色映射
                          s=20,  # 点的大小
                          alpha=0.7)  # 透明度

    # 新增：添加图例
    plt.legend(proxies,
               [str(label) for label in unique_labels],
               loc='upper right',
               title='Class Labels',
               framealpha=0.8)

    # 保留原有设置
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_name, bbox_inches='tight')  # 添加bbox_inches避免图例被截断

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