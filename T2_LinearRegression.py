import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#1 Load data
url = 'auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()


#2 Delete nan data
dataset = dataset.dropna()

#3 Originを3つの列に分割し、yesかnoで答えさせる
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

#4 Sprit data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# データの観察（グラフをpng出力）
# pg = sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# pg.savefig('seaborn_pairplot_default.png')
# train_dataset.describe().transpose()

#5 ラベルと特徴量の分離（燃費"MPG"を予測したいパラメータとして取り出す）
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#6 各数値を平均、標準偏差で正規化

#------ここまで下準備-----------

# -----ここから線形回帰--------

#7 ちょっと下準備
horsepower = np.array(train_features['Horsepower']) # Horsepowerってのは馬力のことね

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower) # 馬力データを正規化

#8 モデル定義（単一変数線形回帰のモデルでは、入力の正規化・線形変換の２つを設定）
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1) # units=1は層の数が1個（出力のみ）、活性化関数はデフォルトではNone（Relu(wx+b)やSoftmax(wx+b)ではなく、単にwx+b）
])

horsepower_model.summary()

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


#8.5 学習前に初期のパラメータ確認
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()

plot_horsepower(x, y)

#9 学習
# 学習（fit）させるときはhistory変数にlossの履歴などを保存すると便利
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

# 学習後
y = horsepower_model.predict(x)
plot_horsepower(x, y)