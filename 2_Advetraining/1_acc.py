import torch
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import utils, data, model

import argparse
"   python 1_acc.py --data_choose 0 --model_choose 0 --train_adv '../data_adv/data_model_Train_adversary.h5py' --test_adv '../data_adv/data_model_Test_adversary.h5py' --save_path '../data_model_adv/data_model_adversary.pth'    "
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")  # 
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")  # 
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")  # 
parser.add_argument("--train_adv", type=str, help="../data_adv/data_model_Train_adversary.h5py")  # 
parser.add_argument("--test_adv", type=str, help="../data_adv/data_model_Test_adversary.h5py")  # 
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")  # 
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")  # 
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")  # 
parser.add_argument("--save_path", help="训练后网络参数的保存位置")  # 
opt = parser.parse_args()
if opt.data_choose == 0:  # 
    opt.num_class = 10
else: opt.num_class = 3
print(opt)

device = utils.try_gpu(4)

net = model.get_classification_net(net_choose=opt.model_choose, num_class=opt.num_class, pretrained=True, path=opt.save_path)

net.to(device)

net.eval()

if opt.data_choose == 0:
    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=False, transform=transform)

elif opt.data_choose == 1:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    testset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

elif opt.data_choose == 2:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    testset = data.SeekThermal(root=opt.root_SeekThermal, train=False, transform=transform)

train_iter = data.get_iter(path=opt.train_adv, batch_size=opt.batch_size)
test_iter = data.get_iter(path=opt.test_adv, batch_size=opt.batch_size)
fgsm_iter = data.get_iter(path=opt.test_adv.replace('pgd', 'fgsm'), batch_size=opt.batch_size)
test_iter_ori = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)

acc1 = utils.evaluate_accuracy_gpu(net, train_iter)
acc2 = utils.evaluate_accuracy_gpu(net, test_iter)
acc3 = utils.evaluate_accuracy_gpu(net, test_iter_ori)
acc4 = utils.evaluate_accuracy_gpu(net, fgsm_iter)

print('pgd训练集正确率')
print(acc1)

print('pgd测试集正确率')
print(acc2)

print('原始测试集正确率')
print(acc3)

print('fgsm测试集正确率')
print(acc4)
