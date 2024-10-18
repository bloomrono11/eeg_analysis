from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_cnn3d_model(input_shape):
    """
     A method to create a CNN 3D Model
    :param input_shape: (depth, height, width, input_channel)
    :return: A model with 3 CNN layers
    :rtype: Sequential

    :Example:
    >>> create_cnn3d_model(input_shape=(64, 62, 5, 1))
    """
    model = Sequential()

    # First 3D convolutional layer
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    # Second 3D convolutional layer
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Output layer (for 4 classes: neutral, sad, fear, happy)
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def reshape_eeg_data_for_cnn3d(eeg_data, w_time):
    # Reshape the data to (batch_size, depth, height, width, channels)
    # Add a channel dimension
    eeg_data_reshaped = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[2], eeg_data.shape[1], eeg_data.shape[3], 1)
    if w_time < 64:
        eeg_data_reshaped = pad_sequences(eeg_data_reshaped, maxlen=64, padding='post', dtype='float32')
        # print(f" Original shape: {eeg_data.shape}, padded Data Shape: {eeg_data_reshaped.shape}")
    return eeg_data_reshaped


def create_cnn2d_model(input_shape):
    """
     A method to create a CNN 2D Model
    :param input_shape: (depth, height, width)
    :return: A model with 2 CNN layers
    :rtype: Sequential

    :Example:
    >>> create_cnn2d_model(input_shape=(64, 62, 5))
    """
    # Build the CNN model
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  # Example shape
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer for classification (4 classes for 4 emotions)
    model.add(layers.Dense(4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Model summary
    model.summary()
    return model


# Reshape the data for CNN 2D model
def reshape_eeg_data_for_cnn2d(eeg_data, w_time):
    # Reshape to (batch_size, height, width, channels)
    eeg_data_reshaped = eeg_data.reshape(eeg_data.shape[0], eeg_data.shape[2], eeg_data.shape[1], eeg_data.shape[3])
    if w_time < 64:
        eeg_data_reshaped = pad_sequences(eeg_data_reshaped, maxlen=64, padding='post', dtype='float32')
        # print(f" Original shape: {eeg_data.shape}, padded Data Shape: {eeg_data_reshaped.shape}")
    return eeg_data_reshaped


def create_cnn1d_model(input_shape):
    """
     A method to create a CNN 1D Model
    :param input_shape: (depth, height*width)
    :return: A model with 1 CNN layers
    :rtype: Sequential

    :Example:
    >>> create_cnn1d_model(input_shape=(64, 62*5))
    """
    # Build the CNN model
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Convolutional Layer 2
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer for classification (4 classes for 4 emotions)
    model.add(layers.Dense(4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Model summary
    # model.summary()
    return model


# Reshape the data for CNN 1D model
def reshape_eeg_data_for_cnn1d(eeg_data, w_time):
    """
        This method is to reshape cnn 1d data
        reshape from: x,y,z,w to: x,y w*z e.g.,  (24, 62, 64, 5) =>  (24, 62, 64*5)
        # Pad sequences if W or w_time < 64
        old_data = eeg_data_reshaped
        print(f" Original shape: {old_data.shape}, padded Data Shape: {eeg_data_reshaped.shape}")
    :param eeg_data:
    :param w_time:
    :return:
    """
    eeg_data_reshaped = eeg_data.reshape(eeg_data.shape[0], w_time, eeg_data.shape[1] * eeg_data.shape[3])
    if w_time < 64:
        eeg_data_reshaped = pad_sequences(eeg_data_reshaped, maxlen=64, padding='post', dtype='float32')
    return eeg_data_reshaped