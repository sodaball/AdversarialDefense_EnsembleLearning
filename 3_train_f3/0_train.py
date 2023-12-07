import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import sys
sys.path.append('../')
import model, train, data, utils

import argparse

"   python 0_train.py --data_choose 2 --model_choose 1 --train_adv_1 '../dataX1/SeekThermal_resnet18_Train_pgd.h5py' --train_adv_2 '../dataX2/SeekThermal_resnet18_Train_pgd.h5py' --test_adv_1 '../dataX1/SeekThermal_resnet18_Test_pgd.h5py' --test_adv_2 '../dataX2/SeekThermal_resnet18_Test_pgd.h5py' --save_path_ori '../modelF1/SeekThermal_resnet18.pth' --save_path_adv '../modelF3/SeekThermal_resnet18_pgd.pth'   "
"   python 0_train.py --data_choose 0 --model_choose 0 --train_adv_1 '../data_adv/adv_fgsm_0.03_train.h5py' --train_adv_2 '../data_adv_adv/adv_adv_fgsm_0.03_train.h5py' --test_adv_1 '../data_adv/adv_fgsm_0.03_test.h5py' --test_adv_2 '../data_adv_adv/adv_adv_fgsm_0.03_test.h5py' --save_path_ori '../data_model/data_model.pth' --save_path_adv '../data_model_F3/data_model_F3.pth'   " 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")  # 
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")  # 
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")  # 
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")  # 
parser.add_argument("--train_adv_1", type=str, help="../data_adv/data_model_Train_adversary.h5py")  # 
parser.add_argument("--train_adv_2", type=str, help="../data_adv/data_model_Train_adversary.h5py")  # 
parser.add_argument("--test_adv_1", type=str, help="../data_adv/data_model_Test_adversary.h5py")  # 
parser.add_argument("--test_adv_2", type=str, help="../data_adv/data_model_Test_adversary.h5py")  # 
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")  # 
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")  # 
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")  # 
parser.add_argument("--save_path_ori", help="原始网络参数的保存位置")  # 
parser.add_argument("--save_path_adv", help="对抗网络参数的保存位置")  # 
opt = parser.parse_args()
if opt.data_choose == 0: 
    opt.num_class = 10
else: opt.num_class = 3
print(opt)

device = utils.try_gpu(0)

net = model.get_classification_net(net_choose=opt.model_choose, num_class=opt.num_class, pretrained=True, path=opt.save_path_ori)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss = nn.CrossEntropyLoss()

train_iter = data.get_iter_f3(path1=opt.train_adv_1, path2=opt.train_adv_2, batch_size=opt.batch_size, shuffle=True)  # 训练集是对抗样本训练集
test_iter = data.get_iter_f3(path1=opt.test_adv_1, path2=opt.test_adv_2, batch_size=opt.batch_size, shuffle=False)  # 测试集是对抗样本测试集

if opt.data_choose == 0:
    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=False, transform=transform)

elif opt.data_choose == 1:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    testset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

elif opt.data_choose == 2:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    testset = data.SeekThermal(root=opt.root_SeekThermal, train=False, transform=transform)

test_iter_ori = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)

train.train(net=net,
            train_iter=train_iter, 
            test_iter=test_iter, 
            optimizer=optimizer, 
            loss=loss, 
            num_epochs=opt.epochs, 
            device=device, 
            optimizer_scheduler=optimizer_scheduler, 
            advtrain=True, 
            test_ori_iter=test_iter_ori, 
            save_path=opt.save_path_adv)

