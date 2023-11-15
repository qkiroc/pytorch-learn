# 线性归回示例
import torch
import matplotlib.pyplot as plt

# 设置随机种子，可以保证每次代码中产生的随机数是一样的
torch.manual_seed(10)

lr = 0.1
# 这里的乘10，是对生成的张量每一个数字都乘10，使得随机数变成0-10之间
x = torch.rand(20, 1) * 10
y = 2 * x + (5 + torch.randn(20, 1))

w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(100):
	# 张量相乘
	wx = torch.mul(w, x)
	# 张量相加
	y_pred = torch.add(wx, b)

	loss = (0.5 * (y - y_pred) ** 2).mean()
	# 反向传播，只有loss反向传播之后参数才能求导
	loss.backward()

	# 张量相减
	b.data.sub_(lr * b.grad)
	w.data.sub_(lr * w.grad)

	if iteration % 10 == 0:
		# x和y的散点图
		plt.scatter(x.data.numpy(), y.data.numpy())
		# 预测值的直线
		plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
		plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
		# x、y轴的取值范围
		plt.xlim(1.5, 10)
		plt.ylim(8, 28)
		plt.title('Iteration: {}\nw:{}b:{}'.format(iteration, w.data.numpy(), b.data.numpy()))
		# 让程序暂停0.5s
		plt.pause(0.5)
		# 0维张量转换成数值还可以用loss.item()
		if loss.data.numpy() < 0.05:
			break