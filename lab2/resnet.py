"""
@File ：resnet.py
@Author ：jamie109
@Date ：2023/5/19
"""
import torch
import torch.nn as nn


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
    # 删掉 不实现 resnet50 了
    # # 记录每个残差块的卷积核个数是否有变化
    expansion = 1

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
        # self.downsample = None
        # 下采样方法
        self.flag = flag
        # if flag is True:
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

        out += identity
        out = self.relu(out)
        return out

# class BottleNeck(nn.Module):
#     """
#
#     """
class ResNet(nn.Module):
    """
    resnet
    """
    def __init__(self, block_num=1, num_classes=10):
        super(ResNet, self).__init__()
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
        # self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        #
        #self.fc1 = nn.Linear(, num_classes)
        self.fc1 = nn.Linear(4096, 1000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, 100)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(100, num_classes)


    # def _make_layer(self, block, channel, block_num, stride=1):
    #     """
    #     :param block: ResNet-18＆ResNet-34 或 ResNet-50（未实现）
    #     :param channel: 一个残差结构中 卷积层的卷积核个数
    #     :param block_num: 一个 conv 结构中 残差结构的个数
    #     """
    #     downsample = None
    #     # ResNet-18＆ResNet-34 第一层不会执行
    #     if stride != 1 or self.in_channel != channel * block.expansion:
    #         # 生成下采样函数
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,stride=stride,bias=False),
    #             nn.BatchNorm2d(channel * block.expansion)
    #         )
    #     layers = []
    #     # 传入第一层 block
    #     layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
    #
    #     self.in_channel = channel * block.expansion
    #     # 剩余
    #     for _ in range(1, block_num):
    #         # ResNet-18＆ResNet-34 in_channel 一直是 64
    #         layers.append(block(self.in_channel, channel))
    #
    #     return nn.Sequential(*layers)

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv1(x)
        print("After conv1 shape:", x.shape)
        x = self.bn1(x)
        print("After bn1 shape:", x.shape)
        x = self.relu1(x)
        print("After relu shape:", x.shape)
        x = self.maxpool(x)
        print("After maxpool shape:", x.shape)

        x = self.layer1(x)
        print("After layer1 shape:", x.shape)
        x = self.layer2(x)
        print("After layer2 shape:", x.shape)
        x = self.layer3(x)
        print("After layer3 shape:", x.shape)
        x = self.layer4(x)
        print("After layer4 shape:", x.shape)
        #x = self.fc1(x)
        #print("After fc1 shape:", x.shape)
        x = torch.flatten(x, 1)
        x = self.relu2(self.fc1(x))
        x = self.relu3(self.fc2(x))
        x = self.fc3(x)
        print("After flatten shape:", x.shape)

        return x


def resnet_18():
    res18 = ResNet(num_classes=10)
    # 输入数据
    input_data = torch.randn(1, 3, 32, 32)

    # 前向传播
    output = res18(input_data)
    print("Final output shape:", output.shape)
    print(res18)
    return res18

# def resnet_34():
#     res34 = ResNet(num_classes=10)
#     print(res34)
#     return res34

if __name__ == '__main__':
    resnet_18()
    # resnet_34()