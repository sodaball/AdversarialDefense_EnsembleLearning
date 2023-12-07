0_generate.py 
运行：python 0_generate.py --model_choose 0 --data_choose 0 --if_train True --adversary 'fgsm' --save_path '../data_model/' --img_path '../data_adv/data_model_iftrain_method.h5py'
提供：选择模型，选择数据集及训练集or测试集，选择攻击方法，给定网络参数保存位置，给定生成对抗样本的h5py文件存放位置
返回：在--img_path位置保存生成的对抗样本

1_show.py
运行：python 1_show.py --img_path '../data_adv/data_model_iftrain_method.h5py'
提供：给定对抗样本集保存位置
返还：此对抗样本集的基本信息

2_acc.py
运行：python 2_acc.py --img_path '../data_adv/SeekThermal_resnet18_Train_pgd.h5py' --save_path '../data_model/SeekThermal_resnet18.pth' --model_choose 1 --data_choose 2
提供：选择对抗样本数据集路径，选择网络参数路径，选择模型，选择数据集

pgd攻击：0.03 0.004 *10

resnet18网络：

SeekThermal训练集：0.22512437810945274(0.9982587064676617)
SeekThermal测试集：0.19725738396624473(0.9345991561181435)

vgg16网络：

cifar10训练集：0.06498(0.99666)
cifar10测试集：0.0596(0.8865)

fgsm攻击：0.03

resnet18网络：

SeekThermal测试集：0.4978902953586498

vgg16网络：

cifar10测试集：0.1406
# SeekThermal攻击效果太差，调高参数如下




攻击在原始网络上进行，验证对于原始网络的攻击成功率

(SeekThermal_resnet18)
pgd攻击：
0.07, 0.008, 10
测试集：0.0010548523206751054
训练集: 0.005970149253731343
fgsm攻击：
0.07
测试集：0.21624472573839662

(cifar10_vgg16)
pgd攻击：
0.03 0.004 *10
测试集：0.06498
训练集：0.0596
fgsm攻击：
0.03
测试集：0.1406

(SeekThermal_vgg16)
pgd攻击：
0.07, 0.008, 10
测试集：0.0
训练集: 0.0014925373134328358
fgsm攻击：
0.07
测试集：0.45358649789029537

(cifar10_vgg16)
pgd攻击：
0.03 0.004 *10
测试集：
训练集：
fgsm攻击：
0.03
测试集：




攻击在增强网络上进行，验证对于原始网络的攻击成功率
(SeekThermal_resnet18)
pgd攻击：
0.07, 0.008, 10
验证集在原始网络：0.8227848101265823
验证集在增强网络：0.10970464135021098