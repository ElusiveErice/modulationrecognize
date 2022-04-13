import numpy as np
from matplotlib import pyplot as plt

from model import AWGN_4CLASS
from train import train
from test import test


if __name__ == '__main__':
    model_name = AWGN_4CLASS
    x_name = ['1:1', '1:2', '1:3', '1:4', '2:1', '2:2', '2:3', '2:4', '3:1', '3:2', '3:3', '3:4', '4:1', '4:2', '4:3',
              '4:4']
    predict_error_max_list = np.zeros(16)
    for i in range(10):
        train(model_name)
        test()
