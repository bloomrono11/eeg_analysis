import os
from typing import Any

import numpy as np
from numpy import ndarray, dtype

from scipy.io import loadmat


def get_session_labels() -> list[list[int]]:
    # Define session labels (assuming they correspond to each file)
    session_labels = [
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],  # Session 1
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],  # Session 2
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]  # Session 3
    ]
    return session_labels


def get_fixed_params() -> tuple[list, list, list]:
    """
    Get fixed parameters for Seed4 usage for any model
    For our models we are using eeg_keys for 24 frames with value of de_LDS{i}
    Other possible values are
    :return:
      data_dirs: list file directory paths where seed4 data is stored
      session_labels: list of session labels provided in seed4 directory root in ReadMe.txt
      eeg_keys: list of eeg keys provided in seed4 file
    """
    # Define the directory where your .mat files are stored
    # data_dir = 'data/eeg/eeg_raw_data/1'
    data_dirs = ['data/eeg/eeg_feature_smooth/1',
                 'data/eeg/eeg_feature_smooth/2',
                 'data/eeg/eeg_feature_smooth/3']
    # 'data/eeg/eeg_feature_smooth/2', 'data/eeg/eeg_feature_smooth/3'

    # EEG keys corresponding to the channels in your .mat files
    eeg_keys = [f'de_LDS{i}' for i in range(1, 25)]  # cz_eeg1 to cz_eeg24, tyc_eeg
    return data_dirs, get_session_labels(), eeg_keys


def load_n_classify(session_labels: list[list[int]], eeg_keys: list[str], idx: int, file: str) -> tuple:
    """
       Processes EEG Data for classification task
         Below are commented code for additional output
         # max_length = max([data[key].shape[0] for key in eeg_keys])
         # print(f'data length {len(data)}, Max Length {max_length}')
         # print(f"data keys {data.keys()}, {data['de_LDS1'].shape}")

       Parameters
       ----------
       session_labels : list  Session labels that were provided with Seed 4 EEG dataset via Readme.txt
       eeg_keys : the edge key to be used for the EEG dataset
       idx : the index of the session label array, [0, 1 or 2]
       file : the name of the dataset file being processed

       Returns
       --------
       eeg_data : ndarray - the numpy nd array
       session_label : list - the single session array label associated with the filelist

    """
    data = loadmat(file)

    # Initialize an empty list to hold processed EEG data
    eeg_data = []

    # Find the maximum length of the time dimension across all keys
    max_length = 0
    for key in eeg_keys:
        max_length = max(max_length, data[key].squeeze().shape[1])  # Get the second dimension (N)

    # Process EEG data for each key
    for key in eeg_keys:
        channel_data = data[key].squeeze()  # Shape: (62, N, 5)

        # Initialize an array to hold padded or truncated data
        channel_processed = np.zeros((62, max_length, 5))  # Shape: (62, max_length, 5)

        # Pad or truncate the data
        for i in range(62):  # Iterate over each channel
            n_time_points = channel_data[i].shape[0]  # Number of time points in this channel
            if n_time_points < max_length:
                # If shorter, pad with zeros
                channel_processed[i, :n_time_points, :] = channel_data[i]  # Fill in the actual data
            else:
                # If longer, truncate
                channel_processed[i, :, :] = channel_data[i][:max_length, :]  # Truncate to max_length

        # Append the processed channel data to the list
        eeg_data.append(channel_processed)

    # Convert the list to a numpy array if needed
    eeg_data = np.array(eeg_data)

    return eeg_data, session_labels[idx]


def load_n_merge_raw_data(dir_path: str) -> ndarray[Any, dtype[Any]]:
    """
       This method is to load each session that from a given directory
       Next remove the xx_eeg{dd} 2 characters xx at the start
       Rename the keys to eeg_1, ... eeg_24
       Find the max sample for the 24 trials, so we can pad the data accordingly
       Iterate the 24 trials, create the dimension of 24 and pad data where necessary
       Finally, merge the data into a nd array where x=15,y=24 and z=62
       Parameters
       ----------
       dir_path : the directory path where EEG raw dataset is stored

       Returns
       --------
       eeg_data_merged : ndarray - the numpy nd array
    """

    eeg_data_per_session = []
    for filename in os.listdir(dir_path):
        # if not filename.endswith('.mat'):
        #    continue

        file_path = os.path.join(dir_path, filename)
        data = loadmat(file_path)
        eeg_signals = {key: data[key] for key in data if '_eeg' in key}

        # Rename the keys to 'eeg_1', 'eeg_2', etc.
        renamed_signals = {f'eeg_{i + 1}': signal for i, (key, signal) in enumerate(eeg_signals.items())}

        max_samples = max(renamed_signals[f'eeg_{i + 1}'].shape[1] for i in range(24))

        # 24 trials per subject are stored in the file
        subject_data = []
        # Extract EEG data
        for i in range(24):  # Loop through all trials
            trial_data = renamed_signals[f'eeg_{i + 1}']
            pad_trial_data = np.pad(trial_data,
                                    ((0, 0),
                                     (0, max_samples - trial_data.shape[1])),
                                    mode='constant')
            subject_data.append(pad_trial_data)

        eeg_data_per_session.append(np.array(subject_data))
        break

    return np.array(eeg_data_per_session)


def main():
    # For EEG CNN 2D with TL
    eeg_data = load_n_merge_raw_data('../data/eeg/eeg_raw_data/1')
    print(f"eeg_data.shape: {eeg_data.shape}")


if __name__ == '__main__':
    main()
