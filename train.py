import tensorflow.keras as keras

from read_data import read_train_data
from model import Model, AWGN_4CLASS


def train(name=AWGN_4CLASS, epochs=10):
    model = Model(name)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error", metrics=['acc'])

    x_train, y_train = read_train_data()

    model.fit(x_train, y_train, epochs=epochs, batch_size=100, validation_split=0.1, shuffle=True, )
    model.save_weights(name + '.h5')
