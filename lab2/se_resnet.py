"""
@File ：se_resnet.py
@Author ：jamie109
@Date ：2023/5/21
"""
import torch
import torch.nn as nn
from resnet import train, validate
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.se_avgpool = nn.AdaptiveAvgPool2d(1)
        self.se_relu = nn.ReLU()
        self.se_fc1 = nn.Linear(in_channel, in_channel // reduction_ratio)
        self.se_fc2 = nn.Linear(in_channel // reduction_ratio, in_channel)

    def forward(self, x):
        batch, channel, height, width = x.size()
        out = self.se_avgpool(x).view(batch, channel)
        # print(f"    @SEBlock after se_avg_pool {out.shape}")
        out = self.se_fc1(out)
        # print(f"    @SEBlock after se_fc1 {out.shape}")
        out = self.se_relu(out)
        out = torch.sigmoid(self.se_fc2(out))
        # print(f"    @SEBlock after se_fc2 {out.shape}")
        out = out.view(batch, channel, 1, 1)
        final_out = x * out
        # print(f"    @SEBlock final_out {final_out.shape}")
        return final_out


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicBlock(nn.Module):
    """
    ResNet-18、ResNet-34
    """
    def __init__(self, in_channel, out_channel, stride=1, flag=False):
        """
        :param flag: 下采样
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # conv2的 in_channels 应该是 out_channel！！
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.seblock = SEBlock(out_channel)
        # self.downsample = None
        # 下采样方法
        self.flag = flag
        self.downsample = None
        if self.flag is True:
            self.downsample = Downsample(in_channel, out_channel)

    # def my_downsample(self, x, d_in_channel, d_out_channel):
    #     self.ds_conv = nn.Conv2d(d_in_channel, d_out_channel, kernel_size=(1, 1), stride=(2, 2), padding=0)
    #     self.ds_batch_normal = nn.BatchNorm2d(d_out_channel)

    def forward(self, x):
        # shortcut分支
        identity = x
        if self.flag is True:
            identity = self.downsample(x)
        # res block
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SEblock
        out = self.seblock(out)

        out += identity
        out = self.relu(out)
        return out

class SEResNet(nn.Module):
    """
    SE-resnet
    """
    def __init__(self, block_num=1, num_classes=10):
        super(SEResNet, self).__init__()
        # 输入特征矩阵的深度
        # self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1,
                               padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet
        self.layer1 = BasicBlock(64, 64, 1)
        self.layer2 = BasicBlock(64, 128, 2, True)
        self.layer3 = BasicBlock(128, 128, 1)
        self.layer4 = BasicBlock(128, 256, 2, True)

        #self.fc1 = nn.Linear(, num_classes)
        self.fc1 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.conv1(x)
        # print("After conv1 shape:", x.shape)
        x = self.bn1(x)
        #print("After bn1 shape:", x.shape)
        x = self.relu1(x)
        #print("After relu shape:", x.shape)
        x = self.maxpool(x)
        # print("After maxpool shape:", x.shape)

        x = self.layer1(x)
        # print("After layer1 shape:", x.shape)
        x = self.layer2(x)
        # print("After layer2 shape:", x.shape)
        x = self.layer3(x)
        # print("After layer3 shape:", x.shape)
        x = self.layer4(x)
        # print("After layer4 shape:", x.shape)
        #x = self.fc1(x)
        #print("After fc1 shape:", x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # print("After fc1 shape:", x.shape)
        #print("After flatten shape:", x.shape)
        return x


def test():
    net = SEResNet()
    # 输入数据
    input_data = torch.randn(1, 3, 32, 32)

    # 前向传播
    output = net(input_data)
    print(net)

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

    net = SEResNet()

    # 定义损失函数，交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 10

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, net)
        validate(lossv, accv, net)

    # 损失
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title('validation loss')
    # 准确率
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title('validation accuracy')