import sys
sys.path.append('../')

import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import h5py

import utils, model, data


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def pgd(model, device, test_loader, epsilon, iter_epsilon, iter_num ):
    
    adv_examples = []

    for i in range(3):
        model[i].to(device)
        model[i].eval()

    # Loop over all examples in test set
    counter = 0
    for img, target in test_loader:

        img, target = img.to(device), target.to(device)

        data = img + torch.tensor(np.random.uniform(-iter_epsilon, iter_epsilon, img.shape), dtype=img.dtype, device=img.device)  # 初始随机噪声
        data = torch.clamp(data, 0, 1)
        data = utils.where(data > img+epsilon, img+epsilon, data)  # 把一个判断语句转换为一个计算
        data = utils.where(data < img-epsilon, img-epsilon, data)  # 确定data在以img为圆心的pgd球内
        
        for num in range(iter_num):
            i = num % 3

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = model[i](data)  
            # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                
            # 官方这里认为已经错分的模型没必要再生成对抗样本了
            # # If the initial prediction is wrong, dont bother attacking, just move on
            # if init_pred.item() != target.item():
                # continue

            # Calculate the loss
            loss = F.cross_entropy(output, target)

            # Zero all existing gradients
            model[i].zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data / 3.

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, iter_epsilon, data_grad)

            perturbed_data = utils.where(perturbed_data > img+epsilon, img+epsilon, perturbed_data)  # 把一个判断语句转换为一个计算
            perturbed_data = utils.where(perturbed_data < img-epsilon, img-epsilon, perturbed_data)

            data = perturbed_data.detach()
        adv_ex = data.squeeze().permute(1, 2, 0).cpu().numpy()
        adv_examples.append((target.item(), adv_ex))
        
        # 计数
        counter += 1
        print(counter)

        # 测试
        # if counter == 100:
        #     break

    # Return the accuracy and an adversarial example
    return adv_examples


model1 = model.get_classification_net(net_choose=1, num_class=3, pretrained=True, path='../modelF1/SeekThermal_resnet18.pth')
model2 = model.get_classification_net(net_choose=1, num_class=3, pretrained=True, path='../modelF2/SeekThermal_resnet18_pgd.pth')
model3 = model.get_classification_net(net_choose=0, num_class=3, pretrained=True, path='../modelF1/SeekThermal_vgg16.pth')
model_list = [model1, model2, model3]

transform  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = data.SeekThermal(root='../data', train=False, transform=transform)
data_iter = torch.utils.data.DataLoader(dataset, batch_size=1,
                                        shuffle=True)

device = utils.try_gpu(0)

adv_list = pgd(model_list, device, data_iter, 0.07, 0.008, 10)

with h5py.File('../dataX4/SeekThermal_resnet18_pgd_2.h5py', 'w') as file_adv:
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for label, image in adv_list:
        # label是一个int，image是一个numpy数组(32, 32, 3), float32的数组 
        file_adv.create_dataset(f'{label}/' + str(counter[label]) + '.jpg',
                                data=image, compression='gzip', compression_opts=9)
        counter[label] = counter[label] + 1
        print(counter)
