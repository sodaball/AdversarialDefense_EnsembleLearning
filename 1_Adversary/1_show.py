from ctypes import util
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import data, utils

import matplotlib.pyplot as plt
import argparse

"   python 1_show.py --img_path '../data_adv/SeekThermal_resnet18_Test_pgd.h5py'   "
parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, help="    '../data_adv/数据集_网络_训练/测试_攻击方法.h5py'    ")
parser.add_argument("--img_choose", type=int, default=0, help="选择图片")
opt = parser.parse_args()
print(opt)

dataset = data.H5py_to_datase(opt.img_path)

print('数据集数量')
print(len(dataset))

print('数据集集图片尺寸及数据类型')
print(dataset[opt.img_choose][0].shape)
print(dataset[opt.img_choose][0].dtype)
print('数据集标签类型')
print(type(dataset[opt.img_choose][1]))

print('数据集数据展示')
print(dataset[opt.img_choose])

to_img = transforms.ToPILImage()
print('数据集打印显示')
plt.imshow(to_img(dataset[opt.img_choose][0]))
plt.savefig('./test.png')
plt.show()

# to_img = transforms.Compose([transforms.ToPILImage(),utils.AddPepperNoise(0.9)])  # 椒盐噪声
# to_img = transforms.Compose([transforms.ToPILImage(),transforms.GaussianBlur((1,1))])  # 高斯滤波
# to_img = transforms.Compose([utils.AddRandomNoise(0.07),utils.BilateralFilter(5, 20),transforms.ToPILImage()])
# print('数据集打印显示')
# plt.imshow(to_img(dataset[opt.img_choose][0]))
# plt.savefig('./test_Pepper.png')
# plt.show()
