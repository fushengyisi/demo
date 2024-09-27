# 画二维坐标图
# 读取csv并作图
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

setting = "xlabel-true"
pred = np.load('./results/'+setting+'/pred.npy')
true = np.load('./results/'+setting+'/true.npy')
real_prediction = np.load('./results/'+setting+'/real_prediction.npy')
metrics = np.load('./results/'+setting+'/metrics.npy')

#[samples, pred_len, dimensions]
print(pred.shape)
print(true.shape)
print(real_prediction.shape)

print(metrics)
# 修改后的打印结果
for i in range(real_prediction.shape[1]):
    print(real_prediction[0, i, 0])





#print(true[0,:,-1])

# draw prediction
plt.figure()
plt.plot(true[0,:,-1], label='true')
plt.plot(pred[0,:,-1], label='pred')

plt.ylim(0,true[0,:,-1].max())
plt.legend()
plt.show()
