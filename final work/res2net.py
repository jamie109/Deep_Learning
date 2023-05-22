import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['Res2Net', 'res2net50']
# 模型链接
model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


class Bottle2neck(nn.Module):
    # 最后一个卷积层的输出通道数相对于输入通道数的扩展倍数
    # 经过一个 Bottle2neck 输出应该是设定的 planes 的4倍。planes * self.expansion
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality 输入的 channel 数目
            planes: output channel dimensionality 输出的 channel 数目
            stride: conv stride. Replaces pooling layer.默认步长为 1，就不会有数据损失。下面的卷积步长都是 1
            downsample: None when stride = 1 用于匹配维度
            baseWidth: basic width of conv3x3 用来控制每个组中输入的 channel 数目，26是哪里来的？
            scale: number of scale. 尺度（scale）维度 s 把上一层的输出分成多少组
            stype: 'normal': normal set. 'stage': first block of a new stage.两种模式：stage normal
            stage不用加上前一小块的输出结果，而是直接 sp = spx[i]
            normal需要加上前一个小块儿的输出，即 sp = sp + spx[i] 论文中讲的
        """
        super(Bottle2neck, self).__init__()
        # Width/baseWidth is just used to control the channel number in each split.
        # We just follow the previous works such as Res2NeXt to use this code style.
        width = int(math.floor(planes * (baseWidth / 64.0)))
        # 1、第一次卷积
        # width：经过第一次卷积输出的 channel 分成 scale 组，每组有 width 个 channel
        # n = s*w
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        # 2、分组
        # nums 是小卷积层的数目 因为 x1 不处理 所以当使用 res2net 时，卷积 = s - 1
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        # stage 模式有个平均池化
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        # 添加 nums 个小卷积。w 个输入 w个输出 每次卷积后都要 bn
        convs = []
        # 在每次卷积后用了 bn 再 relu
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # 3、1*1 卷积进行信息融合
        # 但为什么输出 planes * self.expansion 个 channel，不应该是一开始的 planes 个吗？fixed
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        # 1、第一次卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 2、res2Net 模块
        #  tensor 的第 1 维度，(N, C, H, W) 中的 Channel，分成 w 组
        spx = torch.split(out, self.width, 1)
        """
        是最后一块没卷积。假如 s = 4，这个循环遍历的是 nums = s -1。0、1、2。
        0、1、2 都卷积过了，3 没卷积。出了循环再把 3 加进去。
        """
        for i in range(self.nums):
            # i == 0 是 nums 为 1的情况，sp 就是前面输出的所有，不会发生累加情况
            # stype == ‘stage’模式 不用加上前一小块的输出结果。我应该不会用到它。复现的时候直接删掉这个参数？
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            # 加上前一小块
            else:
                # （i >= 1）sp 是上一小块 conv bn relu 的输出结果 这里把输出跟小块加起来
                sp = sp + spx[i] # 这里是不是应该在进入循环前加一个 sp = None？不用
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            # 下面是输出部分，等循环结束 out 就是全部输出
            # 第一小块不卷积 或者 nums 为 1，输出就是没卷积过的 sp
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        # 最后一块不处理 直接加到结果中
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        # resnet 的 shortcut 分支跟主分支形状不同，需要下采样处理
        # residual 和 x 是一个东西
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """
    Res2Net 的基本残差模块
    """
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        """
        :param block: Res2Net 基本块类，Bottle2neck
        :param layers: Res2Net 每个 ResBlock 有几层
        :param baseWidth:
        :param scale: 尺度 s
        :param num_classes: 分多少类
        """
        self.inplanes = 64 # 输入的 channel？？？
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale # 尺度s
        # 输入 channel 为 3，输出为 64，步长2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 进入 Res2Net 此时 in_channel 为 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 池化 全连接 分类
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 卷积层和批归一化层的参数将得到适当的初始值，有助于网络的训练和收敛
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        跟据输入channel参数，和 blocks，构建一个 res2net模块。
        在第一个块中，对输出channel进行了扩展，扩展了 block.expansion倍。更新 inchannel 参数（inplanes），使输入输出相等
        生成一个 Res2Net layer
        :param block: Bottle2neck 类
        :param planes:
        :param blocks: 多少个 基本res2net块。
        :param stride: 默认步长
        :return:
        """
        downsample = None
        # shape 不同，需要下采样。使输出的 channel 大小为 planes * block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 复现时，stage 模式应该改成 normal 模式。或者直接去掉 stage。
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        # 在一个 layer 之后，输入跟输出就相等了，最终的输出是 planes * block.expansion
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # 这里 self.inplanes = planes * block.expansion，无需下采样。res2net的前一层输入跟输出可以直接相加作为后一层输入。
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

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

        x = self.avgpool(x)
        # 一维
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        # 加载预训练的权重
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())