import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import MNIST_MLP

# 超参数
EPOCHS = 5        # 总共训练5轮
BATCH_SIZE = 64
LR = 1e-3         # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载数据
train_loader, test_loader = get_dataloader(BATCH_SIZE)

# 2. 初始化模型、损失函数、优化器
model = MNIST_MLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 测试集计算准确率函数
def test_acc():
    model.eval()  # 评估模式，关闭dropout等
    total_correct = 0
    total_num = 0
    with torch.no_grad():  # 测试不需要计算梯度，节省资源
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(imgs)
            pred_label = torch.argmax(pred, dim=1)
            total_correct += (pred_label == labels).sum().item()
            total_num += imgs.shape[0]
    acc = total_correct / total_num
    return acc

# 3. 开始训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()   # 清空梯度
        output = model(imgs)    # 前向传播
        loss = criterion(output, labels) # 计算损失
        loss.backward()         # 反向传播求梯度
        optimizer.step()        # 更新权重
        total_loss += loss.item()

    # 每轮跑完测试准确率
    acc = test_acc()
    print(f"第{epoch+1}轮 | 训练损失:{total_loss:.2f} | 测试准确率:{acc:.4f}")

# 4. 训练全部结束，保存训练好的模型权重
torch.save(model.state_dict(), "./mnist_mlp_model.pth")
print("模型已保存：mnist_mlp_model.pth")