"""
@File ：densenet.py
@Author ：jamie109
@Date ：2023/5/20
"""
import torch
import torch.nn as nn
from resnet import train, validate
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class DenseLayer(nn.Module):
    """
    DenseBlock 的一层 输出通道数是 in_channel + growth
    """
    def __init__(self, in_channel, growth):
        super(DenseLayer, self).__init__()
        # 1*1
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, growth * 4, kernel_size=1, stride=1, padding=0, bias=False)
        # 2*2
        self.bn2 = nn.BatchNorm2d(growth * 4)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(growth * 4, growth, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # print(f"    @DenseLayer the shape of x is {x.shape}")
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        # print(f"    @DenseLayer after conv1 is {out.shape}")
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        # print(f"    @DenseLayer after conv1 is {out.shape}")
        final_out = torch.cat((x, out), 1)
        # print(f"    @DenseLayer final_out is {final_out.shape}")
        return final_out


class DenseBlock(nn.Module):
    """
    Dense Block
    """
    def __init__(self, in_channel, growth, layer_num=2):
        super(DenseBlock, self).__init__()
        # 一个 denseblock ：2 denselayer
        self.layer1 = DenseLayer(in_channel, growth)
        self.layer2 = DenseLayer(in_channel + growth, growth)

        # for i in range(layer_num):
        #     layer_name = "layer"

    def forward(self, x):
        # print(f"  @DenseBlock start shape {x.shape}")
        out = self.layer1(x)
        # print(f"  @DenseBlock after layer1 {out.shape}")
        out = self.layer2(out)
        # print(f"  @DenseBlock after layer2 {out.shape}")
        return out


class Transition(nn.Module):
    def __init__(self, in_channel):
        super(Transition, self).__init__()
        self.bn_mid = nn.BatchNorm2d(in_channel)
        self.relu_mid = nn.ReLU()
        self.conv_mid = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, bias=False)
        self.avgpool_mid = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn_mid(x)
        out = self.relu_mid(out)
        out = self.conv_mid(out)
        out = self.avgpool_mid(out)
        return out

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        # prepare
        self.conv0 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        # into denseblock
        self.db1 = DenseBlock(32, 32)
        self.transt = Transition(96)
        self.db2 = DenseBlock(64, 32)

        # 全连接
        self.fc1 = nn.Linear(1152, 400)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        # print(f"@DenseNet the shape of x is {x.shape}")
        out = self.conv0(x)
        # print(f"@DenseNet after conv0 {out.shape}")
        out = self.bn0(out)
        # print(f"@DenseNet after bn0 {out.shape}")
        out = self.relu0(out)
        # print(f"@DenseNet after relu0 {out.shape}")
        out = self.maxpool(out)
        # print(f"@DenseNet after maxpool {out.shape}")
        out = self.db1(out)
        out = self.transt(out)
        out = self.db2(out)
        # print(f"@DenseNet after denseblock2 the shape is {out.shape}")
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.fc2(out)
        # print(f"@DenseNet final out is {out.shape}")
        return out

    #
    # def forward(self):
    #     pass

def test():
    densenet = DenseNet()
    # 输入数据
    input_data = torch.randn(1, 3, 32, 32)

    # 前向传播
    output = densenet(input_data)
    print(densenet)

if __name__ == '__main__':
    # test()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 每次测试或小批量训练的样本数为 4
    batch_size = 4

    # 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # shuffle=True表示在每个epoch训练之前对数据进行洗牌 num_workers=2表示使用2个子进程来加载数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    densenet = DenseNet()

    # 定义损失函数，交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)

    epochs = 10

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, densenet)
        validate(lossv, accv, densenet)

    # 损失
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title('validation loss')
    # 准确率
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title('validation accuracy')