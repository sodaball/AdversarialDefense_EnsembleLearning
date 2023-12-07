import torch
from torchvision import transforms
import torchvision
import sys
sys.path.append('../')
import model, data, utils
import argparse
parser = argparse.ArgumentParser()

"   python 1_acc.py --data_choose 0 --model_choose 0 "
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")  # 
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")  # 
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")  # 
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")  # 
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")  # 
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")  # 
parser.add_argument("--save_path", help="训练后网络参数的保存位置")  # 

opt = parser.parse_args()
if opt.data_choose == 0:  # 
    opt.num_class = 10
else: opt.num_class = 3

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

# 超参数
batch_size = 100


# 设备
device = utils.try_gpu(0)


# 三种对抗样本
# X1 = data.get_iter('../dataX1/SeekThermal_resnet18_Test_pgd.h5py', batch_size=batch_size)
# X2 = data.get_iter('../dataX2/SeekThermal_resnet18_Test_pgd.h5py', batch_size=batch_size)
# X3 = data.get_iter('../dataX1/SeekThermal_vgg16_Test_pgd.h5py', batch_size=batch_size)
# X4 = data.get_iter('../dataX4/SeekThermal_resnet18_pgd_2.h5py', batch_size=100)

X1 = data.get_iter('../data_adv/adv_fgsm_0.03_test.h5py', batch_size=batch_size)


# 测试集X

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=False, transform=transform)

# transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# testset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

# transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# testset = data.SeekThermal(root='../data', train=False, transform=transform)

X = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# 三生网络

# F1 = '../modelF1/SeekThermal_resnet18.pth'
# F2 = '../modelF2/SeekThermal_resnet18_pgd.pth'
# F3 = '../modelF1/SeekThermal_vgg16.pth'

F1 = '../data_model/data_model.pth'
F2 = '../data_model_adv/data_model_adv.pth'
F3 = '../data_model_F3/data_model_F3.pth'

# 网络
net = model.get_triple_net(net_choose=0, num_class=10, pretrained=True, path1=F1, path2=F2, path3=F3).to(device)




# print('X1的正确率')
# print(utils.evaluate_accuracy_gpu(net, X1, device))

# print('X2的正确率')
# print(utils.evaluate_accuracy_gpu(net, X2, device))

# print('X3的正确率')
# print(utils.evaluate_accuracy_gpu(net, X3))

print('test干净样本的正确率')
print(utils.evaluate_accuracy_gpu(net, X, device))

print('test对抗样本的正确率')
print(utils.evaluate_accuracy_gpu(net, X1, device))

# print('X的正确率')
# print(utils.evaluate_accuracy_gpu(net, X, device))
