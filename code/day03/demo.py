# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from matplotlib import pyplot as plt  # 数据可视化库

# 生成合成数据
torch.manual_seed(42)  # 设置随机种子保证结果可复现
X = torch.rand(100, 1) * 10  # 生成100个0到10之间的随机数作为输入特征
y = 2 * X + 3 + torch.randn(100, 1)  # 生成带有噪声的线性关系数据(斜率为2，截距为3)

# 定义带有自定义激活函数的线性回归模型
class CustomActivationModel(nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()  # 调用父类初始化方法
        self.linear = nn.Linear(1, 1)  # 定义单输入单输出的线性层

    # 自定义激活函数
    def custom_activation(self, x):
        return torch.tanh(x) + x  # 使用tanh激活函数并加上原始输入(残差连接)

    # 前向传播
    def forward(self, x):
        return self.custom_activation(self.linear(x))  # 先通过线性层再应用自定义激活函数

# 初始化模型、损失函数和优化器
model = CustomActivationModel()  # 创建模型实例
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器，学习率0.01

# 训练循环
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    # 前向传播
    predictions = model(X)  # 获取模型预测
    loss = criterion(predictions, y)  # 计算损失

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    # 每100轮打印一次训练进度
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 显示学习到的参数
[w, b] = model.linear.parameters()  # 获取线性层的权重和偏置
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")  # 打印学习到的参数

# 绘制模型拟合结果
plt.figure(figsize=(4, 4))  # 创建4x4大小的图形
plt.scatter(X, y, label='Training Data')  # 绘制训练数据散点图
plt.plot(X, w.item()*X + b.item(), 'r', label='Model Fit')  # 绘制模型拟合直线
plt.legend()  # 显示图例
plt.show()  # 显示图形

# 在新数据上测试模型
X_test = torch.tensor([[4.0], [7.0]])  # 创建测试数据
with torch.no_grad():  # 禁用梯度计算
    predictions = model(X_test)  # 获取模型预测
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")  # 打印预测结果