import array

import numpy as np
import matplotlib.pyplot as plt
import load_n_read as lr
import cnn_d1 as cnd1
import eeg_labels as eel

'''
This sample is taken from
 https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188629
 Tutorial links
 https://www.youtube.com/watch?v=_BdBJOOqMes&list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z&ab_channel=TalhaAnwar
 Code Repo
 https://github.com/talhaanwarch/youtube-tutorials/blob/main/2.2%20EEG%20DL%20Classification.ipynb
'''


def prepare_data(l_freq, h_freq, duration, overlap,
                 h_path='data/health/h*.edf',
                 p_path='data/health/s*.edf') -> array:
    # 'data/sample/h*.edf'
    # 'data/sample/h*.edf'
    healthy_fpaths = lr.load_healthy(h_path)
    unhealthy_fpaths = lr.load_unhealthy(p_path)
    healthy_ctrl_epoch_arr = \
        [lr.read_file_eeg(i, l_freq, h_freq, duration, overlap) for i in healthy_fpaths]
    unhealthy_pat_epoch_arr = \
        [lr.read_file_eeg(i, l_freq, h_freq, duration, overlap) for i in unhealthy_fpaths]

    return eel.create_labels(healthy_ctrl_epoch_arr, unhealthy_pat_epoch_arr)


def main():
    # load_healthy()
    # load_unhealthy()

    # file_paths = load_healthy()
    # read_file_eeg(file_paths[0])

    # data = prepare_data(0.5, 45, 5, 1)
    # data_arr = data[0]
    # print(f'feature array shape {features_arr.shape}')
    # ml.perform_ml_with_stats(data_arr, data[1], data[2])

    data = prepare_data(0.1, 60, 5, 1)
    # 'data/sample/h*.edf', 'data/sample/s*.edf')

    data[0] = np.moveaxis(data[0], 1, 2)
    print(f'data_arr shape: {data[0].shape}')

    # cnd1.perform_cnn()
    cnd1.cnn_accuracy(data[0], data[1], data[2])


if __name__ == '__main__':
    main()
