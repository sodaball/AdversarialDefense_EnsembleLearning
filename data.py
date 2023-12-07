import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import h5py
import os


class H5py_to_datase(Dataset):  # 检验见data_1.py
    """
    将h5py文件处理为可以使用DataLoader进行
    标签是h5py的group
    数据是每个group名下存储的数据
    """
    def __init__(self, file, transform=None):
        self.file_object = h5py.File(file, 'r')
        self.transform = transform
        count = 0
        for group in self.file_object:  # 计算数据集总量
            dataset = self.file_object[group]
            count += len(dataset)
        self.count_all = count  # 记录数据集总量
        print(f'共计读取{self.count_all}组数据')

    def __len__(self):
        return self.count_all

    def __getitem__(self, idx):
        if idx >= self.count_all:  # 索引值大于等于长度报错
            raise IndexError()
        count = 0
        label = -1
        for group in self.file_object:
            label = label + 1
            last_count = count
            count = count + len(self.file_object[group])  # 当前的图片总数
            if idx >= count:
                continue
            idx_ = idx - last_count
            img = np.array(self.file_object[group][str(idx_) + '.jpg']).astype(np.float32)
            img = torch.FloatTensor(img).permute(2, 0, 1)
            # print(img.shape, img.dtype)
            if self.transform:
                img = self.transform(img)
            return img, label


def get_iter(path, batch_size, shuffle=False):  # 检验见data_1.py
    dataset = H5py_to_datase(path)
    data_iter = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
    return data_iter


class H5py_to_datase_f3(Dataset):  # 何并X1 X2训练集用于训练F3
    """
    将h5py文件处理为可以使用DataLoader进行
    标签是h5py的group
    数据是每个group名下存储的数据
    """
    def __init__(self, file1, file2, transform=None):
        self.file_object1 = h5py.File(file1, 'r')
        self.file_object2 = h5py.File(file2, 'r')
        self.transform = transform
        count = 0
        for group in self.file_object1:  # 计算数据集总量
            dataset = self.file_object1[group]
            count += len(dataset)
        self.count1 = count
        for group in self.file_object2:  # 计算数据集总量
            dataset = self.file_object2[group]
            count += len(dataset)
        self.count2 = count - self.count1
        self.count_all = count  # 记录数据集总量
        print(f'共计读取{self.count_all}组数据')

    def __len__(self):
        return self.count_all

    def __getitem__(self, idx):
        if idx >= self.count_all:  # 索引值大于等于长度报错
            raise IndexError()
        if idx >= self.count1:  # 在第二组
            file_object = self.file_object2
            idx = idx - self.count1
        else: 
            file_object = self.file_object1
        count = 0
        label = -1
        for group in file_object:
            label = label + 1
            last_count = count
            count = count + len(file_object[group])  # 当前的图片总数
            if idx >= count:
                continue
            idx_ = idx - last_count
            img = np.array(file_object[group][str(idx_) + '.jpg']).astype(np.float32)
            img = torch.FloatTensor(img).permute(2, 0, 1)
            # print(img.shape, img.dtype)
            if self.transform:
                img = self.transform(img)
            return img, label


def get_iter_f3(path1, path2, batch_size, shuffle=False):  # 检验见data_1.py
    dataset = H5py_to_datase_f3(path1, path2)
    data_iter = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
    return data_iter


def FLIR(root, train, transform):  # 检验见data_0.py
    if train:
        dataset = ImageFolder(root=os.path.join(root, 'FLIR', 'Train'), transform=transform)
    else:
        dataset = ImageFolder(root=os.path.join(root, 'FLIR', 'Test'), transform=transform)
    return dataset


def SeekThermal(root, train, transform):  # 检验见data_0.py
    if train:
        dataset = ImageFolder(root=os.path.join(root, 'SeekThermal', 'Train'), transform=transform)
    else:
        dataset = ImageFolder(root=os.path.join(root, 'SeekThermal', 'Test'), transform=transform)
    return dataset


class Reclassification(Dataset):  # 检验见data_2.py
    def __init__(self, trainset, trainset_adv, net, device):
        self.device = device
        self.net = net.to(device).eval()
        self.trainset = trainset
        self.trainset_adv = trainset_adv
        self.count_all = len(self.trainset) + len(self.trainset_adv)

    def __len__(self):
        return self.count_all

    def __getitem__(self, idx):
        if idx >= self.count_all:  # 索引值大于等于长度报错
            raise IndexError()
        if idx >= len(self.trainset):
            idx = int(idx - len(self.trainset))
            trainset = self.trainset_adv
        else:
            trainset = self.trainset
        img, label = trainset[idx]
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.net(img).squeeze(0)
        return output, label
        

def get_reclassification_iter(trainset, trainset_adv, net, device, batch_size, shuffle):  # 检验见data_2.py
    dataset = Reclassification(trainset, trainset_adv, net, device)
    data_iter = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
    return data_iter





# 鲁棒性训练的想法：
# 即直接把边界进行拟合会怎么样 = = 
class Infrared_Box(Dataset):  #
    """
    将h5py文件处理为可以使用DataLoader进行
    标签是h5py的group
    数据是每个group名下存储的数据
    """
    def __init__(self, file, num_class, epsilon):
        self.file_object = h5py.File(file, 'r')
        # self.box_num = num_boundary
        self.box_boundary = epsilon
        count = 0
        for group in self.file_object:  # 计算数据集总量
            dataset = self.file_object[group]
            count += len(dataset)
        self.count = count
        self.count_all = count * (1 + self.box_num * 2)  # 记录数据集总量
        print(f'共计读取{self.count_all}组数据')

    def __len__(self):
        return self.count_all

    def __getitem__(self, idx):
        if idx >= self.count_all:  # 索引值大于等于长度报错
            raise IndexError()
        # 使用idx决定图片在那个boundary上
        count = 0
        label = -1
        for group in self.file_object:
            label = label + 1
            last_count = count
            count = count + len(self.file_object[group])  # 当前的图片总数
            if idx >= count:
                continue
            idx_ = idx - last_count
            img = np.array(self.file_object[group][str(idx_) + '.jpg']).astype(np.float32)
            img = torch.FloatTensor(img).permute(2, 0, 1)
            # print(img.shape, img.dtype)
            return img, label

if __name__ == '__main__':
    pass
