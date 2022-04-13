from tensorflow.keras import layers
from tensorflow.keras import Sequential

AWGN_4CLASS = 'awgn_4class'
AWGN_8CLASS = 'awgn_8class'
AWGN_FREQ_OFFSET_4CLASS = 'awgn_FreqOffset_4class'
AWGN_PHASE_JITTER_4CLASS = 'awgn_PhaseJitter_4class'
AWGN_PHASE_OFFSET_4CLASS = 'awgn_PhaseOffset_4class'
RAYLEIGH_4CLASS = 'rayleigh_4class'

model1 = Sequential([
    layers.Conv2D(filters=64, kernel_size=(2, 4), activation="relu", input_shape=(2, 100, 1)),

    layers.Conv2D(filters=16, kernel_size=(1, 4), activation="relu"),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model2 = Sequential([
    layers.Conv2D(filters=64, kernel_size=(2, 4), activation="relu", input_shape=(2, 100, 1)),

    layers.Conv2D(filters=16, kernel_size=(1, 4), activation="relu"),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="softmax")
])


def Model(name=AWGN_4CLASS):
    if name is AWGN_4CLASS:
        return model1
    elif name is AWGN_8CLASS:
        return model2
