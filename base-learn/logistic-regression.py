# 逻辑回归: 逻辑回归是一种解决二分类问题的模型
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(10)

# 生成数据
sample_nums = 100  # 设置每一类的样本数为100
mean_value = 1.7  # 设置数据的均值，用于生成正态分布中的数据点
bias = 1  # 设置偏置项，用于生成正态分布中的数据点
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias  # 生成属于类别0的数据的坐标
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-mean_value * n_data, 1) + bias  # 生成属于类别1的数据的坐标
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)


class LR(nn.Module):
	def __init__(self):
		super(LR, self).__init__()
		self.linear = nn.Linear(2, 1)
		# 这里使用sigmoid函数是因为逻辑回归的输出值是0到1之间的浮点数，并不是激活函数
		self.sigmoid = nn.Sigmoid()
		self.loss = nn.BCELoss()

	def forward(self, x):
		x = self.linear(x)
		x = self.sigmoid(x)
		return x


lr_net = LR()

lr = 0.01
# 优化器, 随机梯度下降。 momentum这是动量（momentum）的参数，用于加速梯度下降
# SGD在执行step的时候会自动执行梯度清零的操作，所以不需要手动清零
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

for iteration in range(1000):
	y_pred = lr_net(train_x)
	loss = lr_net.loss(y_pred.squeeze(), train_y)
	loss.backward()
	optimizer.step()

	if iteration % 100 == 0:
		mask = y_pred.ge(0.5).float().squeeze()
		correct = (mask == train_y).sum()  # 计算正确数量
		acc = correct.item() / train_y.size(0)  # 计算正确率

		plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
		plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

		# 表示模型中线性层（Linear Layer）的权重参数
		w0, w1 = lr_net.linear.weight.data[0]
		w0, w1 = float(w0.item()), float(w1.item())

		# 表示模型中线性层（Linear Layer）的偏置参数
		plot_b = float(lr_net.linear.bias[0].item())
		plot_x = np.arange(-6, 6, 0.1)
		plot_y = (-w0 * plot_x - plot_b) / w1

		plt.xlim(-5, 7)
		plt.ylim(-5, 7)
		plt.plot(plot_x, plot_y)

		plt.text(-5, 5, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
		plt.title('iteration={}\n w0={:.2f} w1={:.2f} b={:.2f} accuracy={:.2f}'.format(iteration, w0, w1, plot_b, acc))
		plt.legend()

		plt.show()
		plt.pause(0.5)

		if acc > 0.99:
			break