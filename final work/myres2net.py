"""
@File ：myres2net.py
@Author ：jamie109
@Date ：2023/5/22
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.optim as optim

import data


class Downsample(nn.Module):
    """
    下采样
    """
    def __init__(self, in_channel, out_channel, stride=1):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Bottle2neck(nn.Module):
    """
    Res2Net BasicBlock
    """
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 baseWidth=26, scale=4, flag=False):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param stride: 步长，默认为1
        :param downsample: 下采样函数，用于匹配维度（channel）
        :param baseWidth: 用来控制每个组中输入的 channel 数目，参照原始code，默认26
        :param scale: 维度，把上一层的输出分成多少组
        :param flag: 是否需要下采样
        """
        super(Bottle2neck, self).__init__()
        # 有了 in_channel 和 scale 应该就可以控制 width 了，不明白原代码为什么要有这一行
        # in_channel ！= scale * width
        # in_channel 是第一次卷积的输入，要分块的应该是其输出，所以这里要确定 width 是为了控制 conv1 的输出通道
        width = int(math.floor(in_channel * (baseWidth / 64.0)))
        self.downsample = None
        self.scale = scale
        self.width = width
        self.in_channel = in_channel
        self.out_channel = out_channel
        conv1_out_channel = width * self.scale
        # 步长不为1时，处理过的块跟未处理过的块不能直接相加。但对处理过的块上采样会有信息损失，对未处理块下采样，图像尺寸会越卷越小。
        self.stride = stride
        # 对 x 下采样，使 x 的输出 channel 与经过 small conv 的相同
        self.flag = flag
        if self.flag is True:
            self.downsample = Downsample(in_channel, out_channel, stride)

        # step1 1*1卷积
        self.conv1 = nn.Conv2d(in_channel, conv1_out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        # step2 分组 small convs
        # small conv 数目 conv_num = scale - 1 当s为1，conv_num也为1
        self.conv_num = 1 if self.scale == 1 else self.scale - 1
        convs = []
        bns = []
        relus = []
        for i in range(self.conv_num):
            # tmp_conv = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            # tmp_bn = nn.BatchNorm2d(width)
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
            relus.append(nn.ReLU(inplace=True))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relus = nn.ModuleList(relus)
        # if stride != 1:
        #     conv3_out_C =
        # step3 1*1 conv 进行信息融合
        self.conv3 = nn.Conv2d(conv1_out_channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1*1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        print(f"-----Bottle2neck after conv1 is {out.shape}")
        # small convs
        # (N, C, H, W) 中的 Channel，分成 w 组
        spx = torch.split(tensor=out, split_size_or_sections=self.width, dim=1)
        #print(f"spx shape is {spx.shape}")
        print(f"spx[0] shape is {spx[0].shape}")

        for i in range(self.conv_num):
            if i == 0:
                sp = spx[0]
            else:
                # i >= 1 sp 是上一小块 conv bn relu 的输出结果
                # 把输出跟当前未处理小块加起来
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relus[i](sp)
            print(f"sp shape is {sp.shape}")
            # 卷积过的 spx[0]
            if i == 0:
                out = sp
            else:
                out = torch.cat(tensors=(out, sp), dim=1)
        # 最后一块不处理 直接加到结果中
        if self.scale != 1:
            out = torch.cat(tensors=(out, spx[self.conv_num]), dim=1)
        print(f"-----Bottle2neck after small convs is {out.shape}")
        # print(self.convs)
        out = self.conv3(out)
        out = self.bn3(out)
        print(f"-----Bottle2neck after conv3 is {out.shape}")
        # resnet 的 shortcut 分支跟主分支形状不同
        # 对原始输入 x 下采样
        print(f"-----Bottle2neck the origin x is {x.shape}")
        if self.downsample is not None:
            x = self.downsample(x)
        print(f"-----Bottle2neck after ds x is {x.shape}")
        out += x
        out = self.relu3(out)
        print(f"-----Bottle2neck after add is {out.shape}")
        return out


class Res2Net(nn.Module):
    def __init__(self, baseWidth=26, scale=4, class_num=100):
        """
        :param baseWIdth: 默认26
        :param scale: 尺度 s
        :param class_num: 分类数量
        """
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale  # 尺度s
        # conv input_channel = 3 out 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Res2Net Blocks
        self.layer1 = Bottle2neck(in_channel=32, out_channel=32, stride=1, baseWidth=self.baseWidth, scale=self.scale, flag=True)
        self.layer2 = Bottle2neck(32, 64, 1, None, self.baseWidth, self.scale, True)
        # self.layer3 = Bottle2neck(64, 64, 1, None, self.baseWidth, self.scale, True)
        # self.layer4 = Bottle2neck(64, 128, 1, None, self.baseWidth, self.scale, True)

        self.fc1 = nn.Linear(16384, class_num)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv1(x)
        print("After conv1 shape:", x.shape)
        x = self.bn1(x)
        # print("After bn1 shape:", x.shape)
        x = self.relu1(x)
        # print("After relu shape:", x.shape)
        # x = self.maxpool(x)
        # print("After maxpool shape:", x.shape)
        print("==========into res2net blocks==================")
        x = self.layer1(x)
        print("After layer1 shape:", x.shape)
        x = self.layer2(x)
        print("After layer2 shape:", x.shape)
        # x = self.layer3(x)
        # print("After layer3 shape:", x.shape)
        # x = self.layer4(x)
        # print("After layer4 shape:", x.shape)
        x = torch.flatten(x, 1)
        print(f"After 1D is {x.shape}")
        x = self.fc1(x)
        print(f"After fc1 is {x.shape}")
        return x


def test():
    input = torch.randn(1, 3, 32, 32)
    resnet = Res2Net(26, 4, 100)
    out = resnet(input)
    # res2block = Bottle2neck(64, 256, flag=True)
    # out = res2block(input)
    # print(res2block)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # test()
    """
    下载cifar100数据集
    """
    trainloader, testloader, classes = data.get_data()
    """
    展示图片
    """
    batch_size = 8
    # 在训练数据集上创建一个迭代器，用于逐个访问数据集中的样本
    dataiter = iter(trainloader)
    # 从迭代器中获取下一个元素，即包含图像和对应标签的小批量
    images, labels = next(dataiter)
    # show images
    # torchvision.utils.make_grid 函数用于将多个图像合并成一个网格，便于显示多张图像。
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


