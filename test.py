import numpy as np
import matplotlib.pyplot as plt

from read_data import read_test_data
from model import Model, AWGN_4CLASS


def test(name=AWGN_4CLASS):
    model = Model(name)
    model.load_weights(name + '.h5')

    x_test, y_test, EsNoArray = read_test_data()
    y_predict = model.predict(x_test)

    n = y_test.shape[0]
    right = 0
    x_name = ['1:1', '1:2', '1:3', '1:4', '2:1', '2:2', '2:3', '2:4', '3:1', '3:2', '3:3', '3:4', '4:1', '4:2', '4:3',
              '4:4']
    predict_list = np.zeros(16)
    for i in range(n):
        # should be
        axis_1 = int(y_test[i])
        # predict to be
        axis_2 = np.argmax(y_predict[i, :])
        index = (axis_1 % 4) * 4 + axis_2 % 4
        if axis_2 == axis_1:
            right = right + 1
        else:
            predict_list[index] = predict_list[index] + 1

    predict_error_max_index = np.argmax(predict_list)

    with open('predict_result.txt', 'a') as file:
        file.write('predict error max index:' + str(predict_error_max_index))
        predict_result = ''
        for predict in predict_list:
            predict_result = predict_result + ',' + str(predict)
        file.write(predict_result)
        file.write('\n\n')


if __name__ == '__main__':
    print(np.zeros(10))
