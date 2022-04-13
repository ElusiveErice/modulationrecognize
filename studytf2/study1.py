import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow.keras as keras

model = Sequential()
model.add(layers.Dense(4))
model.add(layers.Dense(5))
model.add(layers.Dense(4))

x = tf.ones((100, 3), tf.float32)
y = tf.ones((100, 4), tf.float32)
model.compile(optimizer=keras.optimizers.SGD(0.001), loss="mean_squared_error")

model.fit(x, y, epochs=10, batch_size=10)

model.summary()
test_x = tf.ones((20, 3), tf.float32)
test_y = model.predict(test_x)
print(test_y)
