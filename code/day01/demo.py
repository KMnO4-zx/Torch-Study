# 导入Pytorch库
import torch
# 导入Pytorch的神经网络模块
import torch.nn as nn
# 导入Pytorch的优化器模块
import torch.optim as optim

# 生成合成数据
# 设置随机种子以保证数据可复现
torch.manual_seed(42)
# 创建包含100个0到10之间随机数的张量，形状为(100, 1)
X = torch.rand(100, 1) * 10  
# 创建线性关系的数据，斜率为2，截距为3，并添加噪声
y = 2 * X + 3 + torch.randn(100, 1)  

# 定义用于线性回归的模型类
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # 调用父类初始化
        super(LinearRegressionModel, self).__init__()
        # 初始化一个线性层，输入维度1，输出维度1
        self.linear = nn.Linear(1, 1)  

    def forward(self, x):
        # 前向传播，通过线性层处理输入x返回输出
        return self.linear(x)

# 初始化模型，损失函数，以及优化器
# 创建线性回归模型对象
model = LinearRegressionModel()
# 创建均方误差损失函数对象
criterion = nn.MSELoss()
# 创建随机梯度下降优化器，学习率为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    # 通过模型产生预测值
    predictions = model(X)
    # 计算模型预测值和真实值之间的损失
    loss = criterion(predictions, y)

    # 反向传播和优化
    # 清零优化器的梯度缓存
    optimizer.zero_grad()
    # 反向传播计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

    # 每100个epoch输出当前损失值
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 显示学习到的参数
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# 新数据测试
X_test = torch.tensor([[4.0], [7.0]])
# 关闭梯度追踪
with torch.no_grad():
    # 预测结果
    predictions = model(X_test)
    # 打印预测值
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")