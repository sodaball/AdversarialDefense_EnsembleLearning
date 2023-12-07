from logging import root
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import data, utils, model

path_net = '../modelF2/cifar10_vgg16_pgd.pth'  # 获得网络路径
advset = data.H5py_to_datase('../dataX1/cifar10_vgg16_Test_pgd.h5py')

# cifar10
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform)

# SeekThermal数据集
# transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# trainset = data.SeekThermal(root='../data', train=True, transform=transform)
# testset = data.SeekThermal(root='../data', train=False, transform=transform)

device = utils.try_gpu(7)

net = model.get_classification_net(net_choose=0, num_class=10, pretrained=True, path=path_net)

net.to(device)

net.eval()

print('显示分类向量')
out = torch.softmax(net(advset[3000][0].unsqueeze(0).to(device)),dim=1)
print(out)