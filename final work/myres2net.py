"""
@File ：myres2net.py
@Author ：jamie109
@Date ：2023/5/22
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Bottle2neck(nn.Module):
    """
    Res2Net BasicBlock
    """
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 baseWidth=26, scale=4):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param stride: 步长，默认为1
        :param downsample: 下采样函数，用于匹配维度（channel）
        :param baseWidth: 用来控制每个组中输入的 channel 数目，参照原始code，默认26
        :param scale: 维度，把上一层的输出分成多少组
        """
        super(Bottle2neck, self).__init__()
        # 有了 in_channel 和 scale 应该就可以控制 width 了，不明白原代码为什么要有这一行
        # in_channel ！= scale * width
        # in_channel 是第一次卷积的输入，要分块的应该是其输出，所以这里要确定 width 是为了控制 conv1 的输出通道
        width = int(math.floor(in_channel * (baseWidth / 64.0)))
        self.downsample = downsample
        self.scale = scale
        self.width = width
        self.in_channel = in_channel
        self.out_channel = out_channel
        conv1_out_channel = width * self.scale

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

        # step3 1*1 conv 进行信息融合
        self.conv3 = nn.Conv2d(conv1_out_channel, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv1_out_channel)
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
            # 卷积过的 spx[0]
            if i == 0:
                out = sp
            else:
                out = torch.cat(tensors=(out, sp), dim=1)
        # 最后一块不处理 直接加到结果中
        if self.scale != 1:
            out = torch.cat(tensors=(out, spx[self.conv_num]), dim=1)
        print(f"-----Bottle2neck after small convs is {out.shape}")
        print(self.convs)
        out = self.conv3(out)
        out = self.bn3(out)
        print(f"-----Bottle2neck after conv3 is {out.shape}")
        # resnet 的 shortcut 分支跟主分支形状不同
        # 对原始输入 x 下采样
        if self.downsample is not None:
            x = self.downsample(self.in_channel, self.out_channel)

        out += x
        out = self.relu3(out)
        print(f"-----Bottle2neck after add is {out.shape}")
        return out


class Res2Net(nn.Module):
    pass


def test():
    input = torch.randn(1, 64, 64, 64)
    res2block = Bottle2neck(64, 256)
    out = res2block(input)


if __name__ == '__main__':
    test()

