import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 32
seed = 42

# なんかテキストとラベルを自動でセットにしてくれるらしい
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
              'aclImdb/train', 
              batch_size=batch_size, 
              validation_split=0.2, #trainデータのうち2割を検証用に
              subset='training', 
              seed=seed)

print("===============")

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])
    print("---")

print("=====================")
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])