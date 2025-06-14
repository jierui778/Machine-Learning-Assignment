import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证可复现性
torch.manual_seed(42)

# 超参数
latent_dim = 100
batch_size = 128
image_size = 64
channels = 3
learning_rate = 0.0002
beta1 = 0.5
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 状态尺寸: 512x4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 状态尺寸: 256x8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 状态尺寸: 128x16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 状态尺寸: 64x32x32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 状态尺寸: 3x64x64
        )
    
    def forward(self, input):
        return self.main(input)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入是3x64x64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态尺寸: 64x32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态尺寸: 128x16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态尺寸: 256x8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态尺寸: 512x4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)

# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# 配置数据加载器（这里用CIFAR-10数据集做演示）
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 实际应用中可以下载LSUN数据集或使用自己的数据集
# 这里用CIFAR-10数据集做演示
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 对抗性标签
        real = torch.ones(imgs.size(0), 1, 1, 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, 1, 1, device=device)
        
        # 输入真实图片
        real_imgs = imgs.to(device)
        
        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()
        
        # 生成器输入噪声
        z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
        
        # 生成一批图片
        gen_imgs = generator(z)
        
        # 生成器损失（愚弄判别器的能力）
        g_loss = criterion(discriminator(gen_imgs), real)
        
        g_loss.backward()
        optimizer_G.step()
        
        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()
        
        # 判别器对真实和生成样本的判别能力
        real_loss = criterion(discriminator(real_imgs), real)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # 打印训练进度
        if i % 100 == 0:
            print(f"[第{epoch}轮/{epochs}轮] [第{i}批次/{len(dataloader)}批次] "
                  f"[判别器损失: {d_loss.item():.4f}] [生成器损失: {g_loss.item():.4f}]")
    
    # 每轮结束生成并保存样本图片
    with torch.no_grad():
        sample_z = torch.randn(16, latent_dim, 1, 1, device=device)
        generated = generator(sample_z).cpu()
        
        # 将图片从-1~1缩放到0~1
        generated = 0.5 * generated + 0.5
        
        # 拼接图片并保存
        grid = make_grid(generated, nrow=4)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f"generated_epoch_{epoch}.png")
        plt.close()

# 保存模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')