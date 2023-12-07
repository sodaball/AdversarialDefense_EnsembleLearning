import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tqdm import tqdm


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# 计时功能类
class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


# 数据容器类
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 计算分类结果正确的个数
def accuracy(y_hat, y):  # 验证见utils_0.py
    """Compute the number of correct predictions."""
    # print(y_hat)
    # print(y)
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)  # 此后y_hat为一维向量
    cmp = y_hat.to(y.dtype) == y
    return float(torch.sum(cmp))


# 计算分类结果正确率
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for x, y in tqdm(data_iter):
            x = x.to(device)
            y = y.to(device)
            metric.add(accuracy(net(x).to(device), y), y.numel())
    return metric[0] / metric[1]

# 用于pgd攻击的范围验证函数
def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


# 计算各个类别分类结果的正确率
def evaluate_accuracy_per_class(net, data_iter, num_class, device=None):
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    count = [0] * num_class * 2
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            y_hat = torch.argmax(y_hat, dim=1)
            for i, label in enumerate(y):
                count[label] += 1 # count的前十位记录总数
                if y_hat[i] == label:  # 即判断结果等于标签
                    count[label+num_class] += 1 # count的后十位记录正确个数
            print(count)
    acc = []
    for i in range(num_class):
        acc.append(count[i+num_class] / count[i])
    plt.bar(range(len(acc)), acc)
    # plt.xticks(range(len(acc)))
    plt.xticks(range(len(acc)))
    plt.xlabel('class_code')
    plt.ylabel('accuracy')
    for i in range(len(acc)):
        plt.text(x= i- 0.3 , y=acc[i], s = acc[i])
    plt.legend(loc='best')
    plt.title("cifar10-vgg Single category recognition rate")
    plt.show()


# 检查错分样本的错分类别
def check_per_class(net, data_iter, num_class, device=None):
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    count = [[0]*num_class for i in range(num_class)]
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            y_hat = torch.argmax(y_hat, dim=1)
            for i, label in enumerate(y):
                if y_hat[i] != label:  # 即判断结果不标签
                    count[label][y_hat[i]] += 1 # count的后十位记录正确个数
            print(count)
    
    fig = plt.figure(figsize=(9, 6))
    ax3 = Axes3D(fig)

    for i in range(len(count)):
        ax3.bar(range(len(count[i])), count[i], zs=i, zdir='x', alpha=0.7, width=0.5)

    ax3.set_xlabel('label')
    ax3.set_ylabel('label_hat')
    ax3.set_zlabel('number')
    plt.show()


class AddPepperNoise(torch.nn.Module):
    def __init__(self, snr) -> None:
        super().__init__()
        self.snr = snr

    def forward(self, img):  # img是PIL
        img_ = F.to_tensor(img).permute(1, 2, 0).cpu().detach().numpy() # (3, h, w)
        h, w, c = img_.shape
        signal_pct = self.snr
        noise_pct = (1 - self.snr)
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
        mask = np.repeat(mask, c, axis=2)
        img_[mask == 1] = 1.   # 盐噪声
        img_[mask == 2] = 0.     # 椒噪声
        img_ = F.to_pil_image(torch.tensor(img_, dtype=torch.float32).permute(2, 0, 1))
        return img_


class AddRandomNoise(torch.nn.Module):
    def __init__(self, epsilon) -> None:  
        super().__init__()
        self.epsilon = epsilon

    def forward(self, img):  # img是tensor
        noise = self.epsilon * torch.rand(img.shape)
        img_ = torch.clip(img + noise, 0., 1.)
        return img_

class BilateralFilter(torch.nn.Module):
    def __init__(self, kernal_size, std) -> None:  
        super().__init__()
        self.kernal_size = kernal_size
        self.std = std

    def forward(self, img):  # img是tensor
        img_ = img.permute(1, 2, 0).numpy()  # （h,w,c）
        img_ = cv2.bilateralFilter(img_.copy(), self.kernal_size, self.std, self.std)
        img = torch.tensor(img_).permute(2, 0, 1)  # （c,h,w）
        return img
