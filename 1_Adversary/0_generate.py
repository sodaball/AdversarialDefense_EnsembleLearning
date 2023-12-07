import torch
import torchvision
from torchvision import transforms

import sys 
sys.path.append('../')
import model, utils, adversary, data

import h5py
import argparse

from tqdm import tqdm

"   python 0_generate.py --model_choose 0 --data_choose 0 --if_train True --adversary 'fgsm' --save_path '../data_model/data_model.pth' --img_path '../data_adv/adv_fgsm_0.03_train.h5py'  "
"   python 0_generate.py --model_choose 0 --data_choose 0 --if_train False --adversary 'fgsm' --save_path '../data_model/data_model.pth' --img_path '../data_adv/adv_fgsm_0.03_test.h5py'  "
"   python 0_generate.py --model_choose 0 --data_choose 0 --if_train True --adversary 'fgsm' --save_path '../data_model_adv/data_model_adv.pth' --img_path '../data_adv_adv/adv_adv_fgsm_0.03_train.h5py'  "
"   python 0_generate.py --model_choose 0 --data_choose 0 --if_train False --adversary 'fgsm' --save_path '../data_model_adv/data_model_adv.pth' --img_path '../data_adv_adv/adv_adv_fgsm_0.03_test.h5py'  "
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--model_choose", type=int, help="0代表vgg16网络，1代表resnet18，2代表ViT网络")
parser.add_argument("--data_choose", type=int, help="0代表cifar10数据集，1代表FLIR数据集，2代表SeekThermal数据集")
parser.add_argument("--if_train", type=bool, default=False, help="对训练集or测试集生成对抗样本")
parser.add_argument("--adversary", type=str, help="生成对抗样本的方法")
parser.add_argument("--root_cifar10", type=str, default='../data', help="cifar10数据集路径")
parser.add_argument("--root_FLIR", type=str, default="../data",help="FILR数据集根路径")
parser.add_argument("--root_SeekThermal", type=str, default="../data",help="SeekThermal数据集根路径")
parser.add_argument("--save_path", help="训练后网络参数的保存位置")
parser.add_argument("--img_path", help="对抗样本保存位置")
opt = parser.parse_args()
if opt.data_choose == 0:
    opt.num_class = 10
else: opt.num_class = 3
print(opt)

# 0.gpu
device = utils.try_gpu(0)

# 1. 获取网络
net = model.get_classification_net(net_choose=opt.model_choose, num_class=opt.num_class, pretrained=True, path=opt.save_path)

# 3. 获取测试集
batch_size = opt.batch_size

if opt.if_train:
    if opt.data_choose == 0:
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=True, transform=transform)

    elif opt.data_choose == 1:
        transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        dataset = data.FLIR(root=opt.root_FLIR, train=True, transform=transform)

    elif opt.data_choose == 2:
        transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        dataset = data.SeekThermal(root=opt.root_SeekThermal, train=True, transform=transform)

else:
    if opt.data_choose == 0:
        transform = transforms.Compose(
            [transforms.ToTensor()])

        dataset = torchvision.datasets.CIFAR10(root=opt.root_cifar10, train=False, transform=transform)

    elif opt.data_choose == 1:
        transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        dataset = data.FLIR(root=opt.root_FLIR, train=False, transform=transform)

    elif opt.data_choose == 2:
        transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        dataset = data.SeekThermal(root=opt.root_SeekThermal, train=False, transform=transform)

print('读取数据个数')
print(len(dataset))
data_iter = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                        shuffle=False)

# 4. 生成对抗样本
if opt.adversary == 'fgsm':
    adv_list = adversary.fgsm(net, device, data_iter, 0.03)  # cifar10
    # adv_list = adversary.fgsm(net, device, data_iter, 0.07)  # SeekThermal

elif opt.adversary == 'pgd':
    # adv_list = adversary.pgd(net, device, data_iter, 0.03, 0.004, 10)  # cifar10
    adv_list = adversary.pgd(net, device, data_iter, 0.07, 0.008, 10)  # SeekThermal

# 测试
# print('======')
# print(len(adv_list))
# print(adv_list[0][0])
# print(adv_list[0][1].shape)
# print(adv_list[0][1].dtype)

# 5. 生成h5py文件
with h5py.File(opt.img_path, 'w') as file_adv:
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for label, image in tqdm(adv_list):
        # label是一个int，image是一个numpy数组(32, 32, 3), float32的数组 
        file_adv.create_dataset(f'{label}/' + str(counter[label]) + '.jpg',
                                data=image, compression='gzip', compression_opts=9)
        counter[label] = counter[label] + 1
        # print(counter)
