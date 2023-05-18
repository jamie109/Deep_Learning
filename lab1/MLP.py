import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

# 定义每个小批量（batch）包含的样本数目
batch_size = 32
# 定义了一个 train_dataset 对象，用于加载 MNIST 数据集的训练集，并将数据转换为张量格式。
# 同时，如果本地没有下载 MNIST 数据集，该函数会自动从 PyTorch 官方网站下载并保存到 ./data 文件夹下。
train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=False, #已经下载过，把True改为false
                               transform=transforms.ToTensor())
# 加载 MNIST 数据集的测试集
validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())
# 数据加载器，将上一步中定义的训练集和验证集转换为可迭代的数据加载器。
# 自动对数据进行批量化、打乱顺序和多线程加载等操作
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)# 是否对数据进行随机打乱操作

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

# 打印一个训练集批次的大小和数据类型
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

# 表示绘制的图形不进行缩放
pltsize=1
# 创建一个新的图形，并指定其大小
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(3):
    # 将每个子图排列在一行中，并设置每个子图的位置
    plt.subplot(1,10,i+1)
    # 关闭图像的坐标轴
    plt.axis('off')
    # 绘制图像
    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
    # 为每个子图设置标题，表示对应的标签类别
    plt.title('Class: '+str(y_train[i].item()))


class Net(nn.Module):
    # 4层网络
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)  # 输入层到隐藏层，增加神经元数量
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(400, 100)  # 隐藏层到隐藏层，增加一层
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 50)  # 隐藏层到隐藏层，减少神经元数量
        self.fc3_drop = nn.Dropout(0.2)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)

        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)

        return F.log_softmax(self.fc4(x), dim=1)


# 创建一个 Net 类的实例 model，并将其移动到指定的计算设备（如 GPU 或 CPU）上。
model = Net().to(device)
# 这是一个优化器
# 随机梯度下降（SGD）优化器，使用 model.parameters() 获取模型中的可训练参数，并设置学习率为 0.01 和动量为 0.5
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
# 损失,交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 打印网络
# 打印模型的结构，可以看到模型中包含了三个线性层和两个 dropout 层。
# 同时，可以看到模型输出的大小为 10，表示对于每个输入样本，模型会输出 10 个类别的预测概率
print(model)


# 单个 epoch 的训练过程
def train(epoch, log_interval=200):
    # Set model to training mode
    # 训练模式
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        # 优化器中的梯度缓存清零
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        # 对损失函数进行反向传播，计算每个参数的梯度
        loss.backward()

        # Update weights
        optimizer.step()  # w - alpha * dL / dw
        # 每经过 log_interval 个 batch，打印当前 epoch 的进度和损失函数值
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


# 用于在验证集上评估训练好的模型的性能
def validate(loss_vector, accuracy_vector):
    # 将模型设置为评估模式
    model.eval()
    val_loss, correct = 0, 0
    # 循环
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # 损失累加到 val_loss 变量中
        val_loss += criterion(output, target).data.item()
        # 预测
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    # 在循环结束后，计算整个验证集上的平均损失，并将其附加到 loss_vector 列表中。
    # 计算模型在整个验证集上的精度，并将其附加到 accuracy_vector 列表中。
    # 函数最后会打印出平均损失和精度。
    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

# 损失
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')
# 准确率
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy');
