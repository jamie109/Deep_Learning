import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from myres2net import Res2Net
from data import get_data

res2net = Res2Net(26, 1, 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res2net.to(device)
# 定义损失函数，交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res2net.parameters(), lr=0.001, momentum=0.9)
trainloader, testloader, classes = get_data()
# 单个 epoch 的训练过程
def train(epoch):
    # Set model to training mode
    # 训练模式
    res2net.train()

    running_loss = 0.0
    correct = 0  # 正确预测的图片
    total = 0  # 总数
    cur_loss = 0

    # Loop over each batch from the training set
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # move to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = res2net(inputs)
        loss = criterion(outputs, labels)
        # 对损失函数进行反向传播
        loss.backward()
        # Update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        cur_loss += loss.item()
        #         # 计算每轮预测正确的图片及图片总数
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if i % 1000 == 999:  # print every 1000 mini-batches
            print(
                f'Train Epoch:{epoch}  [{i - 999:5d}, {i + 1:5d}] loss: {running_loss / 1000:.3f} accuracy:{correct / total * 100}')
            running_loss = 0.0

# 用于在验证集上评估训练好的模型的性能
def validate(loss_vector, accuracy_vector):
    # 验证阶段不需要计算梯度
    with torch.no_grad():
        # 将模型设置为评估模式
        res2net.eval()
        val_loss, correct = 0, 0
        # 循环
        for data, target in testloader:
            # move to GPU
            data = data.to(device)
            target = target.to(device)
            output = res2net(data)

            # 损失累加到 val_loss 变量中
            val_loss += criterion(output, target).data.item()
            # 预测
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        # 在循环结束后，计算整个验证集上的平均损失，并将其附加到 loss_vector 列表中。
        # 计算模型在整个验证集上的精度，并将其附加到 accuracy_vector 列表中。
        # 函数最后会打印出平均损失和精度。
        val_loss /= len(testloader)
        loss_vector.append(val_loss)

        accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
        accuracy_vector.append(accuracy)

        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(testloader.dataset), accuracy))


epochs = 10
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), lossv)
plt.title('validation loss')
# 准确率
plt.figure(figsize=(5,3))
plt.plot(np.arange(1,epochs+1), accv)
plt.title('validation accuracy')

