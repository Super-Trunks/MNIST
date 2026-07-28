import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST图片：28×28=784个像素点（输入维度784），10个数字分类(0~9)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # x形状：[batch, 1, 28, 28] 先铺平成一维向量
        batch = x.shape[0]
        x = x.view(batch, -1)  # [64,784]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out