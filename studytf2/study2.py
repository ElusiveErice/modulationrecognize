import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow.keras as keras

# 原图大小 w h n    填充 p
# 卷积核大小 x y z  移动步长 s
# 特征图大小  (w + 2p) / s - x + 1,  (h + 2p) / s - y + 1,  z
model = Sequential([
    layers.Conv2D(filters=64, kernel_size=(2, 4), activation="relu", input_shape=(100, 64, 1)),

    layers.Conv2D(filters=16, kernel_size=(1, 4), activation="relu"),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error", metrics=['acc'])

g1 = tf.random.Generator.from_seed(1)
g2 = tf.random.Generator.from_seed(2)
x = g1.normal((100, 64, 64, 3), dtype=tf.float32)
# y = model.predict(x)
# print(y.shape)
y = tf.ones((100, 4), dtype=tf.float32)
model.fit(x, y, epochs=10, batch_size=20)
