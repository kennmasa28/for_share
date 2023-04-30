import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #kerasとtensor.kerasは異なる
(traindata, trainlabel), (testdata, testlabel) = mnist.load_data() #ここでkengo/.keras/data/mnistがあるかどうかチェックする

print(traindata.shape) # 60000×28×28
print(testdata.shape) # 10000×28×28

# # 一つの画像を表示
# plt.contourf(traindata[0], cmap='gray')
# plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(traindata[i], cmap=plt.cm.binary)
    plt.xlabel(trainlabel[i])
plt.show()