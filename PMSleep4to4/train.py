from model_ulit.loss import trainLoss1,trainLoss2, evaLoss
from model_ulit.model import FeatureExtractor,Classifier
from function import cost_time,getModelSize
from sleepdata.dataset import *
import torch
from model_ulit.metric import class_metric
from tqdm import tqdm
from einops import rearrange

@cost_time
def main_train(config, args, best_f1):
    patience_max = 11
    batch_size = config.config["data_loader"]["args"]["batch_size"]

    # 如果是gpu就用第一个gpu，如果是cpu就用cpu
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载数据集，split_num代表第几次划分数据集，split_num=1代表第一次划分数据集，数据分批导入，减少内存占用
    train_dataset = sleppEDF_train(args.np_data_dir, config.config["data_set"]["sleepedf_split"],
                                   config.config["data_set"]["sq_len"], config.config["data_set"]["sq_step"]
                             )
    data_x_v, data_y_v, data_prompt_v,data_y2_v,channels_v, freq_v, type_n_v \
        = val_data_load(args.np_data_dir, config.config["data_set"]["sleepedf_split"]
                           )

    type_n = 5
    Fmodel = FeatureExtractor(3000, train_dataset.freq,
                   len(train_dataset.channels), config.config["model"], type_n).to(device)
    Cmodel = Classifier(config.config["model"], type_n).to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 指定损失函数，优化器，评价指标
    loss_trainf = trainLoss1(type_n,device)
    loss_trainc = trainLoss2(type_n,device)
    foptimizer = torch.optim.AdamW(Fmodel.parameters(), lr=config.config["optimizer"]["args"]["lr"],
                                  weight_decay=config.config["optimizer"]["args"]["weight_decay"])
    coptimizer = torch.optim.AdamW(Cmodel.parameters(), lr=config.config["optimizer"]["args"]["lr"],
                                  weight_decay=config.config["optimizer"]["args"]["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(foptimizer, step_size=8, gamma=0.10)

    #获取模型参数量
    print("model_size:", getModelSize(Fmodel))
    print("model_size:", getModelSize(Cmodel))

    change_ed = 0
    loss_plot = []
    val_loss_plot = []
    min_loss = 100
    patience = patience_max
    # 训练
    for epoch in range(1, int((config.config["trainer"]["epochs"])) + 1):
        Fmodel.train()
        Cmodel.train()
        loss_sum = 0

        # 输出当前epoch和这个epoch的学习率,batch数量
        print("epoch:", epoch, "lr:", foptimizer.param_groups[0]["lr"], "batch_num:", len(train_loader))
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        train_len = len(train_loader)
        for batch_idx, (data, target,p1,y2) in loop:
            # 将数据放到gpu上
            target =  rearrange(target, 'b l d -> (b l) d').to(device).detach()
            # 情空梯度
            foptimizer.zero_grad()
            data = rearrange(data, 'b l n d -> (b l) n d').to(device).detach()
            p1 = rearrange(p1, 'b l n d -> (b l) n d').to(device).detach()
            x = Fmodel(data,p1)

            x1 = rearrange(x, '(b l)  n -> b l n',l = config.config["data_set"]["sq_len"]).to(device)
            coptimizer.zero_grad()
            predict = Cmodel(x1)
            predict = rearrange(predict, 'b l n -> (b l) n')
            loss_c = loss_trainc(predict, target)
            loss = loss_trainf(target, x=x) + loss_c
            loss.backward()
            coptimizer.step()
            foptimizer.step()

            loop.set_description(f'Epoch [{epoch}/{int((config.config["trainer"]["epochs"]))}]')
            loop.set_postfix(loss=loss.item(),  loss_c=loss_c.item())
            loss_sum += loss.item()
            if  batch_idx % ( train_len // 4) == 0 and batch_idx>0 :
                val_loss_mean, MF1, class_F1, best_f1, patience, min_loss= evaluate(Fmodel, data_x_v,
                                                                data_y_v, data_prompt_v,data_y2_v,
                                                                type_n, device,
                                                                best_f1, min_loss, patience,
                                                                patience_max, config, Cmodel, config.config["data_set"]["sq_len"])

        # 计算验证集的损失
        lr_scheduler.step()
        loss_mean = loss_sum / train_dataset.data_len
        loss_plot.append(loss_mean)
        val_loss_plot.append(val_loss_mean)
        with open(config.config["trainer"]["save_dir"] + "/loss/" + str(
                config.config["data_set"]["sleepedf_split"][1]) + "loss.txt", "a") as f:
            if epoch == 1:
                # 写入一行横向作为分割
                f.write("-------------------------新的训练开始了-------------------------------" + "\n")
            f.write("epoch:" + str(epoch) + " loss:" + str(loss_mean) + " val_loss:"
                    + str(val_loss_mean) + "Mf1_val" + str(MF1) +
                    "min_loss" + str(min_loss) + "class_f1" + str(class_F1) + "\n")
            # 如果是最后一个epoch写入loss_plot
            if epoch == int((config.config["trainer"]["epochs"])):
                f.write("loss_plot:" + str(loss_plot) + "\n")
                # 写入val_loss_plot
                f.write("val_loss_plot:" + str(val_loss_plot) + "\n")
        # 每个epoch结束后，打印loss，val_loss,并将val_loss和loss保存到列表中，存入一个txt文件用于画图,小数点后保留4位
        print("\r", "    ", "epoch:", epoch, "loss:", round(loss_mean, 4), "val_loss:", round(val_loss_mean, 4),
              "f1_avelue:", round(MF1, 4),
              "class_f1:", [round(i, 4) for i in class_F1], "\n", "best_f1:", round(best_f1, 4),
              "min_loss:", round(min_loss, 4), "patience:",
              patience)
        # 打印一行横向
        print("--------------------------------------------------------")
        if patience <= 0 and change_ed == 0:
            optimizer = torch.optim.SGD(Fmodel.parameters(), lr=config.config["optimizer"]["args"]["lr"] * 0.2,
                                        weight_decay=config.config["optimizer"]["args"]["weight_decay"],
                                        momentum=0.9)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            print("change optimizer to SGD, and now lr is", optimizer.param_groups[0]["lr"])
            # 写入日志
            with open(config.config["trainer"]["save_dir"] + "/loss/" + str(
                    config.config["data_set"]["sleepedf_split"][1]) + "loss.txt", "a") as f:
                f.write("change optimizer to SGD\n")
            Fmodel.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
                config.config["data_set"]["sleepedf_split"][1]) + "best_modelF.pth"))
            Cmodel.load_state_dict(torch.load(config.config["trainer"]["save_dir"] + "/model/" + str(
                config.config["data_set"]["sleepedf_split"][1]) + "best_modelC.pth"))
            change_ed = 1
            patience_max = patience_max + 2
            patience = patience_max

        if patience <= 0:
            # 如果没有提升就停止训练，并写入最后的loss_plot和val_loss_plot
            with open(config.config["trainer"]["save_dir"] + "/loss/" + str(
                    config.config["data_set"]["sleepedf_split"][1]) + "loss.txt", "a") as f:
                f.write("loss_plot:" + str(loss_plot) + "\n")
                # 写入val_loss_plot
                f.write("val_loss_plot:" + str(val_loss_plot) + "\n")
            break
    return best_f1


def evaluate(model, data_x_v, data_y_v, data_prompt_v,data_y2_v,type_n, device, best_f1, min_loss, patience, patience_max, config, Cmodel,s_len):
    val_loss_sum = 0
    output_list = []
    target_list = []
    loss_val = evaLoss(type_n).to(device)
    with torch.no_grad():
        model.eval()
        Cmodel.eval()
        for i in range(len(data_x_v)):
            for j in range(0,len(data_x_v[i]), 128):
                if j + 128 < len(data_x_v[i]):
                    data, target,p1 = data_x_v[i][j:j + 128], data_y_v[i][j:j + 128].to(device),data_prompt_v[i][j:j + 128]

                else:
                    data, target,p1 = data_x_v[i][-128:], data_y_v[i][j:].to(device),data_prompt_v[i][-128:]
                x = model(data.to(device), p1.to(device))
                x = rearrange(x, '(b l)  n -> b l n',l = s_len).to(device).detach()
                output = Cmodel(x.detach())
                output = rearrange(output, 'b l n -> (b l)  n',l = s_len)
                if j + 128 > len(data_x_v[i]):
                    real_len = len(data_x_v[i]) - j
                    output  = output[-real_len:]
                val_loss = loss_val(output, target)
                if output_list == []:
                    output_list = output
                    target_list = target
                else:
                    output_list = torch.cat((output_list, output), dim=0)
                    target_list = torch.cat((target_list, target), dim=0)
                val_loss_sum += val_loss.item()
        _, _, _, class_F1, _, _ = class_metric(output_list, target_list)

    MF1 = sum(class_F1) / len(class_F1)
    val_loss_mean = val_loss_sum / (len(output_list) // 128)
    # 计算list:class_f1均值
    # 保存损失值最小的模型
    if val_loss_mean < min_loss:
        min_loss = val_loss_mean
        torch.save(model.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_modelF.pth")
        torch.save(Cmodel.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_modelC.pth")
        patience = patience_max
        print("min_loss:", min_loss, "保存完毕")
    if MF1 > best_f1:
        best_f1 = MF1
        torch.save(model.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_model_f1_all_F.pth")
        torch.save(Cmodel.state_dict(),
                   config.config["trainer"]["save_dir"] + "/model/" + str(
                       config.config["data_set"]["sleepedf_split"][1])
                   + "best_model_f1_all_C.pth")
        print("best_f1:", best_f1, "保存完毕", "MF1:", MF1, "class_F1:", class_F1)
        patience = patience_max
    if val_loss_mean > min_loss and MF1 < best_f1:
        patience -= 1
    return val_loss_mean, MF1, class_F1, best_f1, patience, min_loss
