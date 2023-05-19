import torch
import torchvision
import torchvision.transforms as transforms

# 把它转换成 tensor，映射到0到1之间的浮点数 因为一开始读进来可能是numpy的
# transforms.Normalize(mean, std) 表示将图像的每个通道（R、G、B）的像素值分别减去0.5并除以0.5
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 每次测试或小批量训练的样本数为 4
batch_size = 4
# 训练集
# CIFAR10 已经封装到 torch 里了，老师建议自己看一下，代码比较简单，可以读读怎么加载数据
# CIFAR10 每一类有 6000 张图片，CIFAR100 每一类有 600 张图片
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

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 在训练数据集上创建一个迭代器，用于逐个访问数据集中的样本
# get some random training images
dataiter = iter(trainloader)


#images, labels = dataiter.next()
# 需要修改代码 AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute 'next'
# 从迭代器中获取下一个元素，即包含图像和对应标签的小批量
images, labels = next(dataiter)

# show images
# torchvision.utils.make_grid 函数用于将多个图像合并成一个网格，便于显示多张图像。
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里需要改一下，可以在forword打印print（x.size) print(x.shape)看看形状
        self.conv1 = nn.Conv2d(3, 6, 5)  # kernel_size=5, padding=2, stride=1
        # # 池化窗口大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 运行的参数需要改
        # forward 函数：怎么用这些层对它操作
        #         x = self.conv1(x)
        #         print("output shape of conv1:", x.size())
        #         x = self.F.relu(x)

        #         x = self.pool(x)

        # print("Output shape after conv1:", x.size())

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图展平为一维向量
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

import torch.optim as optim
# 定义损失函数，交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练卷积神经网络
# 修改为一个函数 一次训练
# def train(epoch):
loss_v = []
acc_v = []
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0  # 正确预测的图片
    total = 0  # 总数
    cur_loss = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        cur_loss += loss.item()
        #         # 计算每轮预测正确的图片及图片总数
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    #     # 计算平均损失和准确率
    #     average_loss = running_loss / len(trainloader)
    acc_v.append(correct / total * 100)
    loss_v.append(cur_loss / len(trainloader))

#     print(f'Epoch: {epoch + 1}, Average Loss: {average_loss:.5f}, Accuracy: {accuracy:.2f}%')
#     return average_loss, accuracy
# print('Finished Training')
print(loss_v, acc_v)

# 保存训练好的模型 不执行
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
