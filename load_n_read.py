from glob import glob
import mne as eeg
import numpy as np


# first load the efd files
def load_data_verse_files(file_path='data/health/*.edf') -> list[str]:
    files = glob(file_path)
    # print(f'print file list {files} length {len(files)}')
    return files


def load_healthy(file_path: str) -> list[str]:
    return load_data_verse_files(file_path)


def load_unhealthy(file_path: str):
    return load_data_verse_files(file_path)


def read_file_eeg(file_path: str,
                  l_freq: any, h_freq: any,
                  duration: float, overlap: float) -> np.ndarray:
    data = eeg.io.read_raw(file_path, preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=l_freq, h_freq=h_freq)
    # data shape is 10 columns and rows 1000 freq is 100Hz
    epochs = eeg.make_fixed_length_epochs(data, duration=duration, overlap=overlap)
    data = epochs.get_data()
    # shape # no. of epochs, channels, length of signal
    # print(data.shape)
    return data
