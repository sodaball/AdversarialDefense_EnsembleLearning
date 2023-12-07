import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import data

import matplotlib.pyplot as plt
import argparse

"   python 0_show.py --data_choose 0 --img_choose 0  "
parser = argparse.ArgumentParser()
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")
parser.add_argument("--img_choose", type=int, help="选择第几个图片")
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")
opt = parser.parse_args()
print(opt)

if opt.data_choose == 0:
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=False, transform=transform)

elif opt.data_choose == 1:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = data.FLIR(root=opt.root_FLIR, train=True, transform=transform)
    testset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

elif opt.data_choose == 2:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = data.SeekThermal(root=opt.root_SeekThermal, train=True, transform=transform)
    testset = data.SeekThermal(root=opt.root_SeekThermal, train=False, transform=transform)

print('训练集标签')
print(trainset.class_to_idx)
print('测试集标签')
print(testset.class_to_idx)

print('训练集数量')
print(len(trainset))
print('测试集数量')
print(len(testset))

print('训练集集图片尺寸及数据类型')
print(trainset[opt.img_choose][0].shape)
print(trainset[opt.img_choose][0].dtype)
print('训练集标签类型')
print(type(trainset[opt.img_choose][1]))
print('测试集集图片尺寸及数据类型')
print(testset[opt.img_choose][0].shape)
print(testset[opt.img_choose][0].dtype)
print('测试集标签类型')
print(type(testset[opt.img_choose][1]))

print('训练集数据展示')
print(trainset[opt.img_choose])
print('测试集数据展示')
print(testset[opt.img_choose])

to_img = transforms.ToPILImage()
print('训练集打印显示')
plt.imshow(to_img(trainset[opt.img_choose][0]))
plt.show()
print('测试集打印显示')
plt.imshow(to_img(testset[opt.img_choose][0]))
plt.savefig('./test.png')
plt.show()
