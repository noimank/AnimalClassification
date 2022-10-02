import os
from PIL import  Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

#类别名及对应的类别ID字典
lable_map = { "butterfly":0, "cat": 1, "chicken": 2, "cow": 3, "dog": 4, "elephant": 5, "horse": 6, "ragno": 7, "sheep": 8, "squirrel":9}


#定义Animal10分类数据集
class AnimalDataset(Dataset):
    def __init__(self,root_path):
        self.root_path = root_path
        self.train_dict = self.initData()
        #训练时的数据增强策略
        self.trainAugMethod = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        super(AnimalDataset, self).__init__()
    #初始化数据集的相关信息，这是自己添加的，自己以什么方式预处理数据都行，写不写成函数都没关系
    def initData(self):
        img_cls_name = os.listdir(self.root_path)
        train_dict = {}
        img_index = 0
        for cls_name in img_cls_name:
            img_root_path = os.path.join(self.root_path, cls_name)
            imgs_names = os.listdir(img_root_path)
            img_paths = [os.path.join(img_root_path, i) for i in imgs_names]
            for img_path in  img_paths:
                train_dict[img_index] = {"img_path":img_path, "cls_name": cls_name}
                img_index += 1
        return train_dict
    #必须要重写的函数，函数返回数据集的长度
    def __len__(self):
        return len(self.train_dict.keys())
    #必须要重写的函数，定义了每次迭代会得到什么类型的数据，定义的是每一张图片将会被执行的操作，这里是返回的是字典
    def __getitem__(self, index):
        img_msg = self.train_dict[index]
        #读取图片并将其转换为RGB类型，防止数据中出现非3通道彩图的情况
        img_obj = Image.open(img_msg["img_path"], mode="r").convert('RGB')
        #获取该图片类别对应的类别ID
        lable = torch.tensor(lable_map[img_msg["cls_name"]])
        #对图片进行数据增强操作，为模型提供统一的图片尺寸
        img_t = self.trainAugMethod(img_obj)
        #返回数据
        return {"img_tensor":img_t, "cls_name":img_msg["cls_name"], "img_path":img_msg["img_path"], "label": lable}




#获取数据迭代器，就是将上面的数据集封装成迭代器，下面3个函数都是一样的
def getAnimalTrainDataloader(data_root_path, batch_size, shuffle=True):
    datasets = AnimalDataset(data_root_path)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return dataloader
def getAnimalTestDataloader(data_root_path, batch_size, shuffle=True):
    datasets = AnimalDataset(data_root_path)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return dataloader
def getAnimalValDataloader(data_root_path, batch_size, shuffle=True):
    datasets = AnimalDataset(data_root_path)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return dataloader






if __name__ == '__main__':
    # a = {"kk":45, "imn":89}
    # print(a["kk"])
    # i = Image.open(r"D:\TrainData\animals10\raw-img\cat\137.jpeg")
    # print(np.array(i))
    # print("kjl")
    AnimalTrainDataloader = getAnimalTrainDataloader(r"D:\TrainData\animals10", 32)
    for data in AnimalTrainDataloader:
        print(data)
        break





