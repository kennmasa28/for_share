import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #kerasとtensor.kerasは異なる
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 手書き数字データ（28×28）を読み込む
(traindata, trainlabel), (testdata, testlabel) = mnist.load_data()

#正規化
traindata = traindata/255.0
testdata = testdata/255.0

# ここでは二層のレイヤーモデルとする
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # 28×28 のデータを読み込んで784次元ベクトルとして扱う
    tf.keras.layers.Dense(128, activation='relu'), # 最初のレイヤーには128個のノードが存在
    tf.keras.layers.Dense(10) # 二つ目（最後）のレイヤーには、分類クラスの数（0-9）と同じ数だけのノードが存在
])

# モデルのコンパイル
model.compile(optimizer='adam', # オプティマイザー(adamはモーメンタムとRMSPropを組み合わせた方法)
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # 損失関数（交差エントロピーを指定）
              metrics=['accuracy']) # 指標（ここでは正解率）
print('compile is finished')

# トレーニング
model.fit(traindata, trainlabel, epochs=10)

# 検証
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) #modelにはwxの出力のみ。これをsoftmax関数に通してようやく確率となる
predictions = probability_model.predict(testdata) #predict(data)関数はおそらく、入力したdataをもとに確率P1,...,Pkを求め、最も大きいものを判定する関数
predictions[0]

plt.contourf(traindata[0], cmap='gray')
plt.show()