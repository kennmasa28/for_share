import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000] # そのままだと6万個なので1000まで
test_labels = test_labels[:1000] # そのままだと1万個なので1000まで


#plt.contourf(test_images[0], cmap='gray')
#plt.show()

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model


# Create a basic model instance
model = create_model() # 同じアーキテクチャのモデルじゃないと重みは共有できない

# 何も学習してない初期パラメータそのまま
loss, acc = model.evaluate(test_images, test_labels, verbose=2) 
print("***Untrained model, accuracy: {:5.2f}%".format(100 * acc))
p_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
for i in range(10):
    print("画像の表す数字が{:}である確率は{:0.5f}".format(i,p_model.predict(test_images)[0][i]))

print("------------")

# wを学習済みのものに更新
checkpoint_path = "Tutorial/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2) # 今度は保存した学習済み重みで評価している
print("***Restored model, accuracy: {:5.2f}%".format(100 * acc))
p_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
for i in range(10):
    print("画像の表す数字が{:}である確率は{:0.5f}".format(i,p_model.predict(test_images)[0][i]))