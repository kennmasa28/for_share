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
# train_dataset.describe().transpose()[['mean', 'std']] # 各特徴量の平均値と標準偏差を出力
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features)) 
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())