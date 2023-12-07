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

utils.evaluate_accuracy_per_class(net, test_iter, 10, device)
