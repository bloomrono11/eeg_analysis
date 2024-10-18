import random

import numpy as np
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt

'''
# Implementing Feedforward neural networks with Keras and TensorFlow
#   a. Import the necessary packages
#   b. Load the training and testing data (MNIST)
#   c. Define the network architecture using Keras
#   d. Train the model using SGD
#   e. Evaluate the network
#   f. Plot the training loss and accuracy'''


# loading dataset
def sample_network() -> str:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    a = len(x_train)
    print(f'length of training dataset is a [x_train] {a:.3f}')

    b = len(y_train)
    print(f'length of training dataset is b [y_train] {b:.3f}')
    pre_process(x_train)
    pre_process(y_train)
    pre_process(x_test)
    pre_process(y_test)

    model = create_model()
    summarize_model(model)
    compile_model(model)
    history = train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    return predict(history, model, x_test)


# pre-processing data

def pre_process(dataset: int):
    return dataset / 255


# create the model
def create_model() -> Sequential:
    model = keras.Sequential(
        [keras.layers.Flatten(input_shape=(28, 28)),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(10, activation='softmax')])
    return model


# summarize the model
def summarize_model(model: Sequential):
    model.summary()


# compile the model
def compile_model(model: Sequential):
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')


# train the model
def train_model(model: Sequential, x_train, y_train, x_test, y_test) -> any:
    history = model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=10, batch_size=128)
    return history


# evaluate the model
def evaluate_model(model: Sequential, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test accuracy: {test_acc:.3f}')


# Making prediction on new data
def predict(history: any, model: Sequential, x_test: np.ndarray):
    n = random.randint(1, 9999)
    plt.imshow(x_test[n])
    plt.show()
    predicted_value = model.predict(x_test)

    print("Handwritten number in the image is= %d" % np.argmax(predicted_value[n]))

    history.history.keys()

    # graph representing the model’s accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # graph representing the model’s loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    keras_model_path = 'models/keras/sample.keras'
    model.save(keras_model_path)
    return keras_model_path


def reload_model(keras_model_path: str) -> Sequential:
    restored_keras_model = tf.keras.models.load_model(keras_model_path)
    return restored_keras_model
