from network import *
from utils import *
from sklearn.metrics import accuracy_score
from dataset import getAnimalValDataloader



#根据dataloader评价ACC
def eval_one_epoch( data_root_path, ckpt_path, device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )):
    dataloader = getAnimalValDataloader(data_root_path,batch_size=16)
    #定义模型
    model = resnet34(num_classes=10)
    #加载模型权重
    model, ckpt = load_weight2model(model, ckpt_path, device)
    model.to(device)
    model.eval()
    print("best_acc = ",ckpt["best_acc"])
    pred_cls = []
    real_cls = []
    #根据数据迭代器迭代数据
    for data in dataloader:
        train_x = data["img_tensor"].to(device)
        real_cls.extend(data["label"].numpy().tolist())
        pred = model(train_x)
        cls_idx = torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()
        pred_cls.extend(cls_idx)
    #使用sklearn库中的accuracy_score函数计算acc
    acc = accuracy_score(real_cls, pred_cls)
    return acc



#根据图片路径预测图片分类，并将分类结果写在图片上
def eval(imgsDir, ckpt_path, PredImgSaveDir="./output", device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )):
    #实例化测试数据的处理类
    tdp = TestDataProcessor()
    imgPaths = tdp.getTestImgPaths(imgsDir)
    #实例化选择的模型
    # model = resnet50(num_classes=11)
    model = resnet34(num_classes=10)
    model, ckpt =load_weight2model(model, ckpt_path, device)
    model.to(device)
    #不加model.eval（）直接使用模型，出来的结果就是错的
    model.eval()
    for imgPath in imgPaths:
        img_t = tdp.getTestImgTensor(imgPath, device)
        pred = model(img_t)
        p_map = tdp.getProbMap(pred)
        tdp.drawClsNameonImg(imgPath, pred, PredImgSaveDir)






if __name__ == '__main__':
    # eval(r"D:\UserData\Code\Python\lectureCodes\AnimalClassifiction\train2testimg", r"D:\UserData\Code\Python\lectureCodes\AnimalClassifiction\ckpt\checkpoint_resnet34.pth")

    acc = eval_one_epoch(r"D:\TrainData\animals10\val", r"D:\UserData\Code\Python\lectureCodes\AnimalClassifiction\ckpt\checkpoint_resnet34.pth")
    print(acc)

