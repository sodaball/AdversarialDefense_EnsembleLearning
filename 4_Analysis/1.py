import sys
sys.path.append('../')

import model, utils
import torchvision
import torch
from torchvision import transforms

path0 = '../cifar10_vgg.pth'

device = utils.try_gpu(3)

net = model.get_classification_net(net_choose=1, num_class=10, pretrained=True, path=path0)

net.eval()

net.to(device)

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 100

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
test_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

utils.check_per_class(net, test_iter, 10, device)

"""
[0, 3, 16, 3, 8, 2, 0, 4, 40, 5], 
[11, 0, 0, 1, 0, 0, 1, 0, 11, 24], 
[29, 1, 0, 28, 32, 16, 19, 15, 3, 1],
[25, 0, 42, 0, 32, 82, 18, 25, 9, 5],
[6, 1, 32, 22, 0, 17, 6, 26, 0, 1], 
[6, 0, 26, 101, 20, 0, 2, 45, 1, 1],
[5, 0, 21, 41, 15, 6, 0, 4, 8, 3], 
[9, 2, 5, 8, 20, 13, 0, 0, 0, 2],
[24, 5, 4, 4, 2, 0, 1, 4, 0, 3],
[16, 59, 2, 3, 1, 0, 1, 2, 17, 0]]
"""