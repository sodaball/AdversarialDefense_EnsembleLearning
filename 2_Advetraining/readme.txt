0_train.py
运行：python 0_train.py --data_choose 0 --model_choose 0 --train_adv '../data_adv/cifar10_vgg16_Train_pgd.h5py' --test_adv '../data_adv/cifar10_vgg16_Test_pgd.h5py' --save_path_ori '../data_model/cifar10.pth' --save_path_adv '../data_model_adv/cifar10_vgg16_pgd_.pth'
提供：选择数据集，选择模型，提供对抗样本训练集路径，提供对抗样本测试集路径，提供原始网络参数路径，提供对抗网络保存位置

1_acc.py
运行：python 1_acc.py --data_choose 0 --model_choose 0 --train_adv '../data_adv/data_model_Train_adversary.h5py' --test_adv '../data_adv/data_model_Test_adversary.h5py' --save_path '../data_model_adv/data_model_adversary.pth'
提供：选择数据集，选择模型，提供对抗样本训练集路径，提供对抗样本测试集路径，提供对抗网络保存位置

SeekThermal_resnet18_pgd.png：

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss = nn.CrossEntropyLoss()
epochs = 10
batch_size = 8
pgd训练集：0.9922885572139304
pgd测试集：0.9082278481012658
原始测试集：0.9124472573839663
fgsm测试集：0.8987341772151899

cifar10_vgg16_pgd.png:

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss = nn.CrossEntropyLoss()
epochs = 10
batch_size = 8
pgd训练集：0.99644
pgd测试集：0.8152
原始测试集：0.8099
fgsm测试集：0.7427

SeekThermal_vgg16_pgd.png：

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss = nn.CrossEntropyLoss()
epochs = 10
batch_size = 8
pgd训练集：0.9922885572139304
pgd测试集：0.9082278481012658
原始测试集：0.9124472573839663
fgsm测试集：0.8987341772151899