import torch
import torch.nn.functional as F
import numpy as np
import utils
from tqdm import tqdm

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm( model, device, test_loader, epsilon ):

    adv_examples = []

    model.to(device)
    model.eval()

    # Loop over all examples in test set
    counter = 0
    for data, target in tqdm(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)  
        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        # 官方这里认为已经错分的模型没必要再生成对抗样本了
        # # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
            # continue

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        # output = model(perturbed_data)

        # Check for success
        # final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        adv_ex = perturbed_data.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        adv_examples.append((target.item(), adv_ex))

        # 计数
        counter += 1
        # print(counter)

        # 测试
        # if counter == 5:
        #      break

    # Return the accuracy and an adversarial example
    return adv_examples


def pgd( model, device, test_loader, epsilon, iter_epsilon, iter_num ):
    
    adv_examples = []

    model.to(device)
    model.eval()

    # Loop over all examples in test set
    counter = 0
    for img, target in test_loader:

        img, target = img.to(device), target.to(device)

        data = img + torch.tensor(np.random.uniform(-iter_epsilon, iter_epsilon, img.shape), dtype=img.dtype, device=img.device)  # 初始随机噪声
        data = torch.clamp(data, 0, 1)
        data = utils.where(data > img+epsilon, img+epsilon, data)  # 把一个判断语句转换为一个计算
        data = utils.where(data < img-epsilon, img-epsilon, data)  # 确定data在以img为圆心的pgd球内
        
        for _ in range(iter_num):
            

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)  
            # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            # 官方这里认为已经错分的模型没必要再生成对抗样本了
            # # If the initial prediction is wrong, dont bother attacking, just move on
            # if init_pred.item() != target.item():
                # continue

            # Calculate the loss
            loss = F.cross_entropy(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

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
        # if counter == 5:
        #       break

    # Return the accuracy and an adversarial example
    return adv_examples
