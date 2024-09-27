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

# 预测后面5次迭代的位置信息
num_iterations = 5
predicted_positions = []

for i in range(num_iterations):
    # 假设每次迭代的位置变化是随机的
    random_change = np.random.normal(loc=0, scale=0.01, size=3)
    last_position = positions[-1]
    new_position = tuple(np.array(last_position) + random_change)
    predicted_positions.append(new_position)
    positions.append(new_position)

# 打印三维预测位置信息
print("Predicted positions:")
for position in predicted_positions:
    print(position)

# 用三维视图表示预测结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制已知位置点
known_x, known_y, known_z = zip(*positions[:5])
ax.scatter(known_x, known_y, known_z, c='b', marker='o', label='Known positions')

# 绘制预测位置点
pred_x, pred_y, pred_z = zip(*predicted_positions)
ax.scatter(pred_x, pred_y, pred_z, c='r', marker='^', label='Predicted positions')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
