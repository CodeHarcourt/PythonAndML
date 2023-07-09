import pandas as pd
import os
import torch

# 创建数据集
os.makedirs(os.path.join('..', 'test'), exist_ok=True)
data_file = os.path.join('..', 'test', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 三个特征(列)由逗号隔开:房屋数量，巷子类型，房屋价格
    f.write('NA,Pave,127500\n')  # 写入，每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')  # 创建的人工数据集有四行三列
# 读取数据集
data = pd.read_csv(data_file)
print(data)
# 缺失值处理
inputs, outputs = data.iloc[0:, 0:2], data.iloc[0:, 2]
inputs = inputs.fillna(inputs.mean())
# 数据格式转换
inputs = pd.get_dummies(inputs, dummy_na=True)
print(outputs.dtypes)
X = torch.Tensor(inputs.values)
Y = torch.Tensor(outputs)
print(X, Y)