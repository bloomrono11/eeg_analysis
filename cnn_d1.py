import keras.src.layers as ksl
import numpy as np
from keras import Sequential
from keras.src.backend.common.global_state import clear_session

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# Ref https://github.com/talhaanwarch/youtube-tutorials/blob/main/2.2%20EEG%20DL%20Classification.ipynb
def cnn_accuracy(data_arr, label_arr, grp_arr):
    gkf = GroupKFold(n_splits=2)
    accuracies = []

    # original data (5744, 1250, 19)
    # reshape to (5744 x 1250, 19)
    # later revert to original (5744, 1250, 19)
    # print(f'Reshaped value train_features {train_features.shape}
    #   {train_features.reshape(-1, train_features.shape[-1]).shape}')
    for train_idx, value_idx in gkf.split(data_arr, label_arr, groups=grp_arr):
        train_features, train_labels = data_arr[train_idx], label_arr[train_idx]
        value_features, value_labels = data_arr[value_idx], label_arr[value_idx]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(
            train_features.shape)
        value_features = (scaler.transform(value_features.reshape(-1, value_features.shape[-1]))
                          .reshape(value_features.shape))

        model = cnn_model()
        model.fit(train_features, train_labels, epochs=10, batch_size=10,
                  validation_data=(value_features, value_labels))
        accuracies.append(model.evaluate(value_features, value_labels)[1])

    result = np.mean(accuracies)
    print('Accuracy: ', result)


def cnn_model() -> Sequential:
    clear_session()
    model = Sequential()
    model.add(ksl.Conv1D(filters=5, kernel_size=3, strides=1, input_shape=(6250, 19)))  # 1
    model.add(ksl.BatchNormalization())
    model.add(ksl.LeakyReLU())
    model.add(ksl.MaxPooling1D(pool_size=2, strides=2))  # 2
    model.add(ksl.Conv1D(filters=5, kernel_size=3, strides=1))  # 3
    model.add(ksl.LeakyReLU())
    model.add(ksl.MaxPooling1D(pool_size=2, strides=2))  # 4
    model.add(ksl.Dropout(0.5))
    model.add(ksl.Conv1D(filters=5, kernel_size=3, strides=1))  # 5
    model.add(ksl.LeakyReLU())
    model.add(ksl.AveragePooling1D(pool_size=2, strides=2))  # 6
    model.add(ksl.Dropout(0.5))
    model.add(ksl.Conv1D(filters=5, kernel_size=3, strides=1))  # 7
    model.add(ksl.LeakyReLU())
    model.add(ksl.AveragePooling1D(pool_size=2, strides=2))  # 8
    model.add(ksl.Conv1D(filters=5, kernel_size=3, strides=1))  # 9

    model.add(ksl.LeakyReLU())
    model.add(ksl.GlobalMaxPooling1D())  # 10
    model.add(ksl.Dense(1, activation='sigmoid'))  # 11

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def summary(model: Sequential):
    print(model.summary())


def perform_cnn():
    model = cnn_model()
    summary(model)
