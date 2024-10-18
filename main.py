# This is a sample Python script.
import sample_neral_network as snn

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # model_path = snn.sample_network()
    # model_path = 'models/keras/sample.keras'
    # model = snn.reload_model(model_path)
    # print(f'Model loaded: {model}')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
