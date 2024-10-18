import numpy as np


def create_labels(healthy_ctrl_epoch_arr, unhealthy_pat_epoch_arr):
    healthy_label = [len(i) * [0] for i in healthy_ctrl_epoch_arr]
    unhealthy_label = [len(i) * [1] for i in unhealthy_pat_epoch_arr]

    data_list = healthy_ctrl_epoch_arr + unhealthy_pat_epoch_arr
    label_list = healthy_label + unhealthy_label
    group_list = [[i] * len(j) for i, j in enumerate(data_list)]
    data_arr = np.vstack(data_list)
    label_arr = np.hstack(label_list)
    group_arr = np.hstack(group_list)

    print(data_arr.shape, label_arr.shape, group_arr.shape)
    print(f'length of data_arr: {len(data_arr)}')
    # Label is required for classification normal person(0) and patient(1)
    print(f'length of label_arr: {len(label_arr)}')
    # Group is required so that during training and trials data
    # of one individual is not mixed with another individual
    # e.g., Subject A has epoch 0, 1, 2 and Subject B has epoch 0, 1, 2
    # if we group them properly then
    #   during training all of Subject A epochs go to training
    #   during testing  all of Subject B epochs go to testing
    # this ensures no issues with data corruption
    print(f'length of group_arr: {len(group_arr)}')

    return [data_arr, label_arr, group_arr]
