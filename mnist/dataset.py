from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    # 数据预处理：图片转张量 + 归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集专属均值方差
    ])

    # 自动下载MNIST数据集，root="./data" 数据存在mnist/data文件夹
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 打包成批次迭代器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader