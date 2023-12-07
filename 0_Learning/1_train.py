import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')
import utils, data, model, train

import argparse

"   python 1_train.py --data_choose 0 --model_choose 0 --save_path '../data_model/data_model.pth'   "
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")
parser.add_argument("--save_path", help="训练后网络参数的保存位置")
opt = parser.parse_args()
if opt.data_choose == 0:
    opt.num_class = 10
else: opt.num_class = 3
print(opt)

device = utils.try_gpu(0)

if opt.data_choose == 0:
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, download=True, train=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, download=True, train=False, transform=transform)

elif opt.data_choose == 1:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = data.FLIR(root=opt.root_FLIR, train=True, transform=transform)
    testset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

elif opt.data_choose == 2:
    transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = data.SeekThermal(root=opt.root_SeekThermal, train=True, transform=transform)
    testset = data.SeekThermal(root=opt.root_SeekThermal, train=False, transform=transform)

train_iter = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False)

net = model.get_classification_net(net_choose=opt.model_choose, num_class=opt.num_class, pretrained=False)


optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss = nn.CrossEntropyLoss()

train.train(net, train_iter, test_iter, optimizer, loss, opt.epochs, device, optimizer_scheduler)

net.to('cpu')

torch.save(net.state_dict(), opt.save_path)
