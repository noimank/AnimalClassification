import numpy as np
import torch.cuda
from tqdm import tqdm
from dataset import getAnimalTrainDataloader, getAnimalTestDataloader
from network import *
from utils import *
from sklearn.metrics import accuracy_score

train_agrs = {
    #神经网络学习率
    "learning_rate": 1e-3,
    #模型进行迭代的最大次数
    "max_epochs": 120,
    #模型推理使用的设备
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    #图片分类数，需要根据数据集进行调整
    "num_classes":10,
    #训练集的路径
    "data_root_path": r"/data/luyk/TrainData/animals10/train",
    # "data_root_path": r"D:\TrainData\animals10\train",
    #模型训练的批次大小，即模型一次要对多少张图片进行推理计算
    "batch_size": 32,
    #模型训练的断点保存路径
    "checkpoint_save_path": "ckpt/checkpoint_resnet50.pth",
    #每次迭代数据集是否打乱数据的顺序
    "data_shuffle": True,
    #当测试集大于0.99时停止训练
    "test_acc": 0.995

}

#训练一次模型，并计算loss还有acc
def train_one_epoch(model, dataloader, optimizer,loss_fn, train_agrs):
    avg_loss = []
    pred_cls = []
    real_cls = []
    model.train()
    for data in dataloader:
        train_x = data["img_tensor"].to(train_agrs["device"])
        real_cls.extend(data["label"].numpy().tolist())
        train_y = data["label"].to(train_agrs["device"])
        #模型进行前向传播，得到结果
        pred = model(train_x)
        #从结果中计算损失
        loss = loss_fn(pred, train_y)
        #获得分类类别概率最大的类别ID
        cls_index = torch.argmax(pred, dim=1).cpu().numpy().tolist()
        pred_cls.extend(cls_index)
        #清空优化器保存的模型梯度，准备更细模型参数
        optimizer.zero_grad()
        #反向传播，计算梯度
        loss.backward()
        #优化器根据计算的梯度，更新模型的参数
        optimizer.step()
        avg_loss.append(loss.item())
    acc = accuracy_score(real_cls, pred_cls)
    avg_loss = np.array(avg_loss).mean()
    return acc, avg_loss

#评价一次模型的效果，返回acc
def eval_one_epoch(model, dataloader, train_agrs):
    pred_cls = []
    real_cls = []
    model.eval()
    for data in dataloader:
        train_x = data["img_tensor"].to(train_agrs["device"])
        real_cls.extend(data["label"].numpy().tolist())
        pred = model(train_x)
        cls_idx = torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()
        pred_cls.extend(cls_idx)
    acc = accuracy_score(real_cls, pred_cls)
    return acc


def train(train_args):
    #定义神经网络模型
    # model = resnet34(num_classes=train_args["num_classes"]).to(train_args["device"])
    model = resnet50(num_classes=train_args["num_classes"]).to(train_args["device"])
    #定义神经网络的优化器，更新神经网络权重
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["learning_rate"])
    #定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss().to(train_args["device"])
    #定义数据集迭代器
    dataloader = getAnimalTrainDataloader(train_args["data_root_path"], train_agrs["batch_size"], train_args["data_shuffle"])
    testDataloader = getAnimalTestDataloader(train_args["data_root_path"], train_agrs["batch_size"], train_args["data_shuffle"])
    #加载训练断点数据
    model, checkpoint = load_weight2model(model, ckpt_path=train_args["checkpoint_save_path"])
    best_acc = checkpoint["best_acc"]
    for epoch in tqdm(range(train_args["max_epochs"])):
        acc, avg_loss = train_one_epoch(model,dataloader, optimizer, loss_fn, train_agrs)
        test_acc = eval_one_epoch(model, testDataloader, train_agrs)
        if test_acc>best_acc:
            best_acc = test_acc
            checkpoint["best_acc"] = test_acc
            checkpoint["model_weight"] = model.state_dict()
            torch.save(checkpoint, train_args["checkpoint_save_path"])
            if best_acc > train_args['test_acc']:
                print("测试集已超过设定阈值{}，停止训练".format(train_args["test_acc"]))
                break
        print("第{}次迭代结束，本次平均train_loss={}，train_acc={}, test_acc={}".format(epoch+1, avg_loss, acc, test_acc))





if __name__ == '__main__':
    train(train_args=train_agrs)
    # pred = [0,1,1,0,2,0]
    # real_p = [0,1,2,1,2,1]
    # acc = accuracy_score(real_p, pred)
    # print(acc)

