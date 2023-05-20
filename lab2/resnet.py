"""
@File ：resnet.py
@Author ：jamie109
@Date ：2023/5/19
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet-18、ResNet-34
    """
    # 记录每个残差块的卷积核个数是否有变化
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        :param downsample: 下采样函数，特征矩阵维度缩放
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样方法
        self.downsample = downsample

    def forward(self, x):
        # shortcut分支
        identity = x
        if self.downsample is not None:
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
    def __init__(self, block, blocks_num, num_classes=10):
        super(ResNet, self).__init__()
        # 输入特征矩阵的深度
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        # resnet
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        #
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, channel, block_num, stride=1):
        """
        :param block: ResNet-18＆ResNet-34 或 ResNet-50（未实现）
        :param channel: 一个残差结构中 卷积层的卷积核个数
        :param block_num: 一个 conv 结构中 残差结构的个数
        """
        downsample = None
        # ResNet-18＆ResNet-34 第一层不会执行
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 生成下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        # 传入第一层 block
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))

        self.in_channel = channel * block.expansion
        # 剩余
        for _ in range(1, block_num):
            # ResNet-18＆ResNet-34 in_channel 一直是 64
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc1(x)
        return x


def resnet_18():
    res18 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(res18)
    return res18

def resnet_34():
    res34 = ResNet(BasicBlock, [3, 4, 6, 3])
    print(res34)
    return res34

if __name__ == '__main__':
    #resnet_18()
    resnet_34()