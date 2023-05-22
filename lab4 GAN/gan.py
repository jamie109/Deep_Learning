import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


class Discriminator(torch.nn.Module):
    # 判别器 对输入进行处理，并输出一个0到1之间的概率值
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        # LeakyReLU激活函数，具有负斜率为0.2
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        # (batch_size, 1, 28, 28)的张量展平为大小为(batch_size, 784)的张量
        x = x.view(x.size(0), 784) # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        # 输出的概率
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    # 生成器 将输入噪声向量转换为输出图像
    def __init__(self, z_dim=100):
        # z_dim是输入噪声向量的维度
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        # tanh函数进行激活，将其范围限制在[-1, 1]之间
        out = torch.tanh(out) # range [-1, 1]
        # convert to image 
        # 将大小为(batch_size, 784)的张量转换为大小为(batch_size, 1, 28, 28)的张量，表示为图像形式
        out = out.view(out.size(0), 1, 28, 28)
        return out


# 显示图像
def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())

def get_data():
    # let's download the Fashion MNIST data, if you do this locally and you downloaded before,
    # you can change data paths to point to your existing files
    # dataset = torchvision.datasets.MNIST(root='./MNISTdata', ...)
    dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5,), (0.5,))]),
                                                download=True)
    # 每批 64
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataset, dataloader


if __name__ == '__main__':
    # instantiate a Generator and Discriminator according to their class definition.
    D = Discriminator()
    print(D)
    G = Generator()
    print(G)
    dataset, dataloader = get_data()
    # Now let's set up the optimizers
    optimizerD = torch.optim.SGD(D.parameters(), lr=0.01)
    optimizerG = torch.optim.SGD(G.parameters(), lr=0.01)
    # and the BCE criterion which computes the loss above:
    criterion = nn.BCELoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device: ', device)
    # Re-initialize D, G:
    D = Discriminator().to(device)
    G = Generator().to(device)
    # Now let's set up the optimizers (Adam, better than SGD for this)
    optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
    optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
    # optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
    # optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
    lab_real = torch.ones(64, 1, device=device)
    lab_fake = torch.zeros(64, 1, device=device)

    # for logging:
    collect_x_gen = []
    fixed_noise = torch.randn(64, 100, device=device)
    fig = plt.figure()  # keep updating this one
    plt.ion()
    loss_D = []
    loss_G = []
    for epoch in range(10):  # 3 epochs
        for i, data in enumerate(dataloader, 0):
            # STEP 1: Discriminator optimization step
            x_real, _ = next(iter(dataloader))
            x_real = x_real.to(device)
            # reset accumulated gradients from previous iteration
            optimizerD.zero_grad()

            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real)

            z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # STEP 2: Generator optimization step
            # reset accumulated gradients from previous iteration
            optimizerG.zero_grad()

            z = torch.randn(64, 100, device=device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

            lossG.backward()
            optimizerG.step()
            if i % 100 == 0:
                x_gen = G(fixed_noise)
                show_imgs(x_gen, new_fig=False)
                fig.canvas.draw()
                print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                    epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
        # End of epoch

        loss_D.append(float(lossD))
        loss_G.append(float(lossG))

        x_gen = G(fixed_noise)
        collect_x_gen.append(x_gen.detach().clone())

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, 11), loss_G)
        plt.savefig('loss_G')

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(1, 11), loss_D)
        plt.savefig('loss_D')

        # 随机数
        noise = torch.randn(8, 100)
        # 8
        generated_image = G(noise)
        show_imgs(generated_image, new_fig=False)
        # alter
        fixed_noise = noise.repeat(5, 1)
        for i in range(0, 8):
            fixed_noise[i][5] = 0.1
        for i in range(8, 16):
            fixed_noise[i][25] = 0.1
        for i in range(16, 24):
            fixed_noise[i][45] = 0.1
        for i in range(24, 32):
            fixed_noise[i][65] = 0.1
        for i in range(32, 40):
            fixed_noise[i][85] = 0.1
        # fixed_noise
        print(fixed_noise.shape)
        # print(fixed_noise[1])
        generated_image = G(fixed_noise)
        show_imgs(generated_image, new_fig=False)
