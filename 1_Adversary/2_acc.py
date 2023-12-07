import sys
sys.path.append('../')
import data, model, utils

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import argparse

# "   python 2_acc.py --img_path '../dataX3/SeekThermal_resnet18_Test_fgsm.h5py' --save_path '../modelF2/SeekThermal_vgg16.pth' --model_choose 0 --data_choose 2 "
"python 2_acc.py --img_path '../data_adv/adv_fgsm_0.03_test.h5py' --save_path '../data_model/data_model.pth' --model_choose 0 --data_choose 0"
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100, help="size of each image batch")
parser.add_argument("--img_path", type=str, help="选择数据集    '../data_adv/数据集_网络_训练/测试_攻击方法.h5py'    ")
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")
parser.add_argument("--save_path", help="训练后网络参数的保存位置")
opt = parser.parse_args()
if opt.data_choose == 0:
    opt.num_class = 10
else: opt.num_class = 3
print(opt)

device = utils.try_gpu(0)

net = model.get_classification_net(net_choose=opt.model_choose, num_class=opt.num_class, pretrained=True, path=opt.save_path)

net.to(device)

net.eval()

# 验证对抗样本的攻击性
iter = data.get_iter(path=opt.img_path, batch_size=opt.batch_size)

# 添加椒盐噪声后
# transform = transforms.Compose([transforms.ToPILImage(),utils.AddPepperNoise(0.930),transforms.ToTensor()])
# 高斯滤波
# transform = transforms.GaussianBlur((3,3))
# 随机噪声
# transform = utils.AddRandomNoise(0.14)
# 双边滤波
# transform = transforms.Compose([utils.BilateralFilter(10, 20),utils.AddRandomNoise(0.07)])
# dataset = data.H5py_to_datase(opt.img_path, transform=transform)
# iter = DataLoader(dataset=dataset, shuffle=False, batch_size=opt.batch_size)

acc = utils.evaluate_accuracy_gpu(net, iter)

print('正确率')
print(acc)
