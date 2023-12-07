from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import accuracy, evaluate_accuracy_gpu, Timer, Accumulator
import adversary


def train(net, train_iter, test_iter, optimizer, loss, num_epochs, device, optimizer_scheduler=None,
          advtrain=False, test_ori_iter=None, save_path=None):
    train_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    test_accuracy_list_advertrain = []
    print('training on', device)
    net.to(device)
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # 总损失值， 正确个数， 总数
        net.train()
        print(f'epoch{epoch}开始迭代')
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)  # 损失自动取均值
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * x.shape[0], accuracy(y_hat, y), x.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            print(f'loss: {train_l}')
        if advtrain:
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            test_acc_ = evaluate_accuracy_gpu(net, test_ori_iter)
            print(f'epoch{epoch}结束迭代')
            print(f'训练集正确率{train_acc}')
            print(f'对抗样本测试集正确率{test_acc}')
            print(f'原始样本测试集正确率{test_acc_}')
            train_loss_list.append(metric[0]/metric[2])
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
            test_accuracy_list_advertrain.append(test_acc_)
        else:
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            print(f'epoch{epoch}结束迭代')
            print(f'训练集正确率{train_acc}')
            print(f'测试集正确率{test_acc}')
            train_loss_list.append(metric[0]/metric[2])
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
        if optimizer_scheduler:
            optimizer_scheduler.step()
        if save_path:
            net.to('cpu')
            torch.save(net.state_dict(), save_path)
            net.to(device)
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    plt.plot(range(1, num_epochs + 1), train_loss_list, linestyle='-', color='blue', label='train loss')
    plt.plot(range(1, num_epochs + 1), train_accuracy_list, linestyle='--', color='red', label='train acc')
    plt.plot(range(1, num_epochs + 1), test_accuracy_list, linestyle='-.', color='green', label='test acc')
    if advtrain:
        plt.plot(range(1, num_epochs + 1), test_accuracy_list_advertrain, linestyle=':', color='yellow', label='test acc ori')
    plt.xlim(1, num_epochs)
    plt.xticks(np.arange(1, num_epochs + 1, 1))
    plt.ylim(0., 3.)
    plt.yticks(np.arange(0, 3.1, 0.2))
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()


def advtrain(net, train_iter, test_iter, optimizer, loss, num_epochs, device, optimizer_scheduler=None, save_path=None):
    """
    用于对抗训练的训练函数，每次对一个样本生成对抗样本，然后同时计算此样本与对抗样本的梯度后，更新网络参数
    tips1:每次生成对抗样本都是基于当前最新网络生成的对抗样本    
    """
    print('training on', device)
    net.to(device)
    timer, num_batches = Timer(), len(train_iter)
    train_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # 总损失值， 正确个数， 总数
        metric_adv = Accumulator(3)
        net.train()
        print(f'epoch{epoch}开始迭代')
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            # 对x逐个生成对抗样本
            x_adv = torch.zeros_like(x, dtype=torch.float32)
            ori_iter = []
            for k in range(x.shape[0]):  # 人为制造一个数据迭代器
                ori_iter.append((x[k:k+1], y[k:k+1]))
            # 对ori_iter做循环，每次返还二元组，0位置为形状是(1, c, h, w)的原始图片，1位置是形状为(1)的标签
            adv_list = adversary.pgd(net, device, ori_iter, 0.03, 0.004, 10)
            for k, (_, adv_img) in enumerate(adv_list):
                adv_img = torch.tensor(adv_img).permute(2, 0, 1).unsqueeze(0)
                x_adv[k:k+1] = adv_img
            y_adv_hat = net(x_adv)
            l1 = loss(y_hat, y)  # 损失自动取均值
            l2 = loss(y_adv_hat, y)
            l = (l1 + l2) * 0.5  # 对2*batch_size组数据求均值
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l1 * x.shape[0], accuracy(y_hat, y), x.shape[0])
                metric_adv.add(l2 * x.shape[0], accuracy(y_adv_hat, y), x.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_l_adv = metric_adv[0] / metric_adv[2]
            print(f'ori_loss: {train_l}')
            print(f'adv_loss: {train_l_adv}')
        train_acc = metric[1] / metric[2]
        train_acc_adv = metric_adv[1] / metric_adv[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch{epoch}结束迭代')
        print(f'ori_acc{train_acc}')
        print(f'adv_acc{train_acc_adv}')
        print(f'验证集正确率{test_acc}')
        print(f'当前{epoch}epoch')
        if optimizer_scheduler:
            optimizer_scheduler.step()
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')