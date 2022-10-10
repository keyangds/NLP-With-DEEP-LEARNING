## this code uses for plotting average loss curve 

import numpy as np

loss1 = np.load("loss1.npy")

loss2 = np.load("loss2.npy")
loss3 = np.load("loss3.npy")
loss4 = np.load("loss4.npy")
loss5 = np.load("loss5.npy")

loss1 = list(loss1)
loss2 = list(loss2)
loss3 = list(loss3)
loss4 = list(loss4)
loss5 = list(loss5)

step = np.arange(0, len(loss1))

for i in step:
    step[i] = i*1000

print(step)


avg_loss = []

for i in range(0,len(loss1)):
    loss = (loss1[i] + loss2[i] + loss3[i] + loss4[i] + loss5[i])/5
    avg_loss.append(loss)



import matplotlib.pyplot as plt

plt.plot(step, avg_loss)
plt.xlabel("Training Step")
plt.ylabel("Avg Train Loss")
plt.title("Loss Curve")

plt.savefig("loss.png")
