import numpy as np
import scipy.io as sio
from tensorflow.keras import utils

seqLen = 100
nClass = 4


def read_train_data():
    x_data_mat = sio.loadmat('./dataset/train_data.mat')
    x_data_complex = x_data_mat['train_data']
    x_data_real = x_data_complex.real
    x_data_imag = x_data_complex.imag
    x_data_real = x_data_real.reshape((x_data_real.shape[0], seqLen, 1))
    x_data_imag = x_data_imag.reshape((x_data_imag.shape[0], seqLen, 1))
    x_train = np.stack((x_data_real, x_data_imag), axis=1)
    y_data_mat = sio.loadmat('./dataset/train_label.mat')
    y_data = y_data_mat['train_label']
    y_train = utils.to_categorical(y_data, nClass)

    # 打乱数据顺序
    index = np.arange(y_train.shape[0])
    np.random.shuffle(index)
    x_train = x_train[index, :]
    y_train = y_train[index]

    return x_train, y_train


def read_test_data():
    data_mat = sio.loadmat('./dataset/test_data.mat')
    data_complex = data_mat['test_data']
    data_real = data_complex.real
    data_imag = data_complex.imag
    EsNoArray = data_real[:, -1]
    y_test = data_real[:, -2]
    data_real = data_real[:, 0:seqLen]
    data_imag = data_imag[:, 0:seqLen]
    data_real = data_real.reshape((data_real.shape[0], seqLen, 1))
    data_imag = data_imag.reshape((data_imag.shape[0], seqLen, 1))
    x_test = np.stack((data_real, data_imag), axis=1)
    return x_test, y_test, EsNoArray


read_train_data()
