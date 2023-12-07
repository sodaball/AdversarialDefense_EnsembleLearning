0_show.py
运行：python 0_show.py --data_choose 0 --img_choose 0
提供：提供数据集，提供查看的图片序号

1_train.py
运行：python 1_train.py --data_choose 0 --model_choose 0 --save_path '../data_model/data_model.pth'
提供：选择数据集，选择模型，选择训练结果存放位置
返还：在--save_path中存放训练结果


训练过程：
vgg16

cifar10:
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
batch_size = 64
epochs = 10
acc_train = 0.99666
acc_test = 0.8865

SeekThermal:
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
batch_size = 16
epochs = 10
acc_train = 0.9982587064676617
acc_test = 0.9187763713080169

FLIR:


resnet18

cifar10:
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
batch_size = 64
epochs = 10
acc_train = 0.99988
acc_test = 0.8364

SeekThermal:
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
optimizer_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
batch_size = 16
epochs = 10
acc_train = 0.9982587064676617
acc_test = 0.9345991561181435

FLIR:
