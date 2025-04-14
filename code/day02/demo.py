# 导入Pytorch库
import torch
# 导入Pandas库用于数据处理
import pandas as pd

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
# 生成包含100个元素的张量，元素值在0到10之间
X = torch.rand(100, 1) * 10  
# 生成与X线性相关，并加上随机噪声的y值
y = 2 * X + 3 + torch.randn(100, 1)  

# 将生成的数据保存到data.csv文件中
data = torch.cat((X, y), dim=1)  # 将X和y按照列拼接
df = pd.DataFrame(data.numpy(), columns=['X', 'y'])  # 转换为DataFrame
df.to_csv('data.csv', index=False)  # 保存为CSV文件，不保存索引

# 再次导入相关库以避免重复导入
import torch
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 定义线性回归数据集类
class LinearRegressionDataset(Dataset):
    def __init__(self, csv_file):
        # 从CSV文件加载数据
        self.data = pd.read_csv(csv_file)
        # 转换为Pytorch张量，指定数据类型为float32，并调整形状
        self.X = torch.tensor(self.data['X'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(self.data['y'].values, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        # 返回数据集长度
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回指定索引的样本
        return self.X[idx], self.y[idx]

# 使用DataLoader加载训练集
dataset = LinearRegressionDataset('data.csv')  # 创建数据集对象
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 创建数据加载器

# 定义线性回归模型类
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 定义单输入和输出的线性层
     
    def forward(self, x):
        # 定义前向传播过程，直接通过线性层
        return self.linear(x)

# 初始化模型、损失函数和优化器
model = LinearRegressionModel()  # 创建模型实例
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率为0.01

# 训练循环
epochs = 1000  # 指定训练轮数
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)  # 计算损失

        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

    # 每100轮打印一次训练进度
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 显示学习到的参数值
[w, b] = model.linear.parameters()  # 获取权重和偏置参数
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# 在新数据上测试模型
X_test = torch.tensor([[4.0], [7.0]])  # 创建测试数据
with torch.no_grad():  # 关闭梯度计算，提高测试速度
    predictions = model(X_test)  # 计算预测值
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")  # 打印预测结果
