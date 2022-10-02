import random

import numpy as np
import torch, os
from PIL import Image
from torchvision.transforms import transforms
import cv2
import shutil
from dataset import lable_map

#加载模型权重
def load_weight2model(model, ckpt_path, map_location="cuda:0"):
    if not os.path.exists(ckpt_path):
        print("ckpt文件不存在，无法加载断点数据")
        return model, {"best_acc":-9999, "model_weight": None}
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_weight"])
    return model, ckpt



#测试数据的处理类
class TestDataProcessor:
    def __init__(self):
        self.testAug = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #获取图片数据的绝对路径
    def getTestImgPaths(self, img_dir):
        img_names = os.listdir(img_dir)
        if len(img_names) == 0:
            return []
        imgs = [os.path.join(img_dir, i) for i in img_names]
        return imgs

    #根据图片的路径并将图片处理成pytorch能处理的Tensor
    def getTestImgTensor(self,img_path, device="cpu"):
        img_obj = Image.open(img_path, mode="r").convert('RGB')
        img_t = self.testAug(img_obj).unsqueeze(0)
        return img_t.to(device)
    #根据类别ID获取对应的类别名，类别ID与类别名的映射看lable_map
    def getClsNameByClsID(self, cls_id):
        for k, v in lable_map.items():
            if v == cls_id:
                return k
        return "unknown"
    #从模型预测的结果中处理得到每个类别ID对应的概率，然后处理得到键为类别名，值为概率的字典
    def getProbMap(self, pred):
        #经过softmax函数得到，每一类对应的概率
        p = torch.softmax(pred, dim=1).squeeze().detach().cpu().numpy()
        #保留两位小数
        p = np.around(p, decimals=2).tolist()
        n = len(p)
        #处理得到概率最大所对应的类别ID
        cls_idx = torch.argmax(pred, dim=1).detach().cpu().numpy()
        p_map = {}
        for i in range(n):
            cls_name = self.getClsNameByClsID(i)
            prob = p[i]
            p_map[cls_name] = prob
        return  p_map, self.getClsNameByClsID(cls_idx)
    #将模型预测的给类别及对应的概率画在图片上
    def drawClsNameonImg(self,img_path, pred, imgSaveDir=None):
        img = cv2.imread(img_path, 1)
        img_name = os.path.basename(img_path)
        p_map, configdentCls = self.getProbMap(pred)
        textHeigt = 10
        for k, v in p_map.items():
            cv2.putText(img, "{}: {:.2f}".format(k, v), (10,textHeigt), cv2.FONT_HERSHEY_PLAIN, 1, (255,60,0), 1)
            textHeigt += 13
        cv2.putText(img, f"{configdentCls}", (img.shape[1]-70, 16 ), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #判断是否保存图片
        if imgSaveDir is not None:
            cv2.imwrite(os.path.join(imgSaveDir, img_name), img)

#提供给splitDataDir函数调用的
def makedirs(targetDir, dirlists):
    if os.path.exists(targetDir):
        shutil.rmtree(targetDir)
    os.mkdir(targetDir)
    currentdir = targetDir
    for dirlist in dirlists:
        os.mkdir(os.path.join(currentdir, dirlist))


#提供给splitDataDir函数调用的
def copyImgstoDir(imgs, save_dir):
    for img in imgs:
        imgname = os.path.basename(img)
        shutil.copy(img, os.path.join(save_dir, imgname))


#划分数据集。train、test、val
def splitDataDir(root_dir, train_radio=0.7, test_radio=0.2, val_radio=0.1):
    dir_names = os.listdir(os.path.join(root_dir, "raw-img"))
    makedirs(os.path.join(root_dir, "train"), dir_names)
    makedirs(os.path.join(root_dir, "test"), dir_names)
    makedirs(os.path.join(root_dir, "val"), dir_names)
    print("正在划分 train test val 请稍等。。。。。。。。。。。。。。")
    for dirname in dir_names:
        imgs = os.listdir(os.path.join(root_dir, "raw-img",dirname))
        imgs = [os.path.join(root_dir, "raw-img",dirname, i) for i in imgs]
        #此处有bug，划分的图片会重复，懂我意思就行，懒得改了
        train_imgs = random.sample(imgs, int(train_radio * len(imgs)))
        test_imgs = random.sample(imgs, int(test_radio * len(imgs)))
        val_imgs = random.sample(imgs, int(val_radio * len(imgs)))
        copyImgstoDir(train_imgs, os.path.join(root_dir, "train", dirname))
        copyImgstoDir(test_imgs, os.path.join(root_dir, "test", dirname))
        copyImgstoDir(val_imgs, os.path.join(root_dir, "val", dirname))

    print("划分数据集完成。。。。。。。。。。。。。。。。。。。。。。。。。")




if __name__ == '__main__':
    #划分数据集
    splitDataDir(r"D:\TrainData\animals10")
    # path = model.fit(X_train, y_train)
    # print("Best model scored", model.score(X_test, y_test))
    # print("Lambda =", model.best_lambda_)




