import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
"""
CIFAR-100数据集包含100个不同的类别，分为20个粗粒度类别和100个细粒度类别。
CIFAR-100数据集共包含60,000个彩色图像，每个图像的尺寸为32x32像素。
其中，50,000个图像用于训练集，另外10,000个图像用于测试集.
"""
def get_data():
    # 定义数据预处理的转换
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)

    # 加载测试集
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=2)

    # 类别标签名称
    classes = trainset.classes

    # 打印数据集信息
    print('训练集大小:', len(trainset))
    print('测试集大小:', len(testset))
    print('类别数量:', len(classes))

    return trainloader, testloader, classes

# if __name__ == '__main__':
#
#     trainloader, testloader, classes = get_data()



