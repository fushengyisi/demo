import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 已知的前5次迭代的位置信息
positions = [
    [0.35679, 0.20399, 0.57],
    [0.35719, 0.20399, 0.56],
    [0.35957, 0.20960, 0.58],
    [0.35533, 0.20373, 0.57],
    [0.35600, 0.20386, 0.56]

]

# 将位置信息转换为NumPy数组
positions = np.array(positions)

# 计算位置信息的差值
diffs = np.diff(positions, axis=0)

# 计算差值的平均值作为转移矩阵的估计
transition_matrix = np.mean(diffs, axis=0)

# 预测后面5次迭代的位置信息
predicted_positions = []
current_position = positions[-1]
for _ in range(5):
    next_position = current_position + transition_matrix
    predicted_positions.append(tuple(next_position))
    current_position = next_position

# 打印预测的三维位置信息
print("Predicted positions:")
for position in predicted_positions:
    print(position)

# 绘制三维视图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制已知位置信息
ax.scatter(*zip(*positions), c='b', label='Known positions')

# 绘制预测位置信息
ax.scatter(*zip(*predicted_positions), c='r', label='Predicted positions')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
