import torch
import numpy as np

# 创建张量
# 直接创建
'''
torch.tensor(
  data,           数据
  dtype,          数据类型
  device,         所在设备，cuda/cpu
  requires_grad,  是否需要梯度于锁页内存
)
'''
arr = np.ones((3, 3))
print("ndarray的数据类型：", arr.dtype)
t = torch.tensor(arr)
print(t)

# 通过torch.from_numpy创建张量
arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
t1 = torch.from_numpy(arr1)
print(t1)

# 依据数值创建
'''
torch.zeros(
  size,          张量的形状
  out,           输出的张量
  layout,        内存中布局形式
  device,
  requires_grad
)
torch.zeros_like(input)                 根据输入的形状创建
torch.ones()                            创建全1张量
torch.ones_like(input)
torch.full()                            指定数值
torch.full_like(input)
torch.arange(start,end,steps)           等差数列
torch.linspace(start,end,steps)         均分数列
torch.logspace(start,end,steps,base=10) 对数均分数列
torch.eye(n, m)                         单位对角矩阵
torch.normal(mean, std)                 正态分布
torch.randn()                           标准正态分布
'''
out_t = torch.tensor([1])
t2 = torch.zeros((3, 3), out=out_t)

print(t2, '\n', out_t)

# 张量操作
# 张量拼接和切分
# torch.cat()
# torch.stack
# torch.chunk 切分
# torch.split