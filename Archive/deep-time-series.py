import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
tf.__version__, keras.__version__

path = r'/Users/sr2/OneDrive - University College London/PhD/Research/Missions/SWARM/Non-Flight Data/Analysis/Nov-21/data/'

file_name = 'ML-070222.h5'
load_hdf = path + file_name
#load_hdf = pd.read_hdf(load_hdf)
print(load_hdf)

df = pd.DataFrame({'age':    [ 3,  29],
                   'height': [94, 170],
                   'weight': [31, 115]})
print(df)

tf.constant([[1., 2., 3.], [4., 5., 6.]]) # 2x3 matrix
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
#t[..., 1,  tf.newaxis]

#from tensorflow import keras
K = keras.backend
K.square(K.transpose(t)) + 10
t.numpy()
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)

def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = 5.0, 3.0


def df_dw1(w1, w2):
    return 6 * w1 + 2 * w2
def df_dw2(w1, w2):
    return 2 * w1

df_dw1(w1, w2)


w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])

with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
try:
    dz_dw2 = tape.gradient(z, w2)
except RuntimeError as ex:
    print(ex)

def cube(x):
    return x ** 3

cube(2)

tf_cube = tf.function(cube)
#print(tf_cube)


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full.shape

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
# plt.imshow(X_train[0], cmap="binary")
# plt.axis('off')
# plt.show()

y_train
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
X_valid.shape
X_test.shape

# n_rows = 4
# n_cols = 10
# plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
# for row in range(n_rows):
#     for col in range(n_cols):
#         index = n_cols * row + col
#         plt.subplot(n_rows, n_cols, index + 1)
#         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
#         plt.axis('off')
#         plt.title(class_names[y_train[index]], fontsize=12)
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


with open(model_pathfile, 'wb') as file:
    pickle.dump(model, file)


#print(history.params())

# import pandas as pd
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.xlabel("Epoch")
# plt.show()
