import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

from eeg_mat_load import get_session_labels


# Bi HDM

# Bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Bandpass filter (e.g., 0.5–50 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


# Function to pad or truncate EEG trials
def pad_or_truncate_trial(trial, target_length):
    if trial.shape[1] < target_length:  # If trial is shorter, pad it
        padding = np.zeros((trial.shape[0], target_length - trial.shape[1]))
        return np.hstack((trial, padding))
    else:  # If trial is longer, truncate it
        return trial[:, :target_length]


# Load and preprocess EEG data for each subject and session
# Load and preprocess EEG data with variable trial lengths
def load_and_preprocess_eeg_data(data_dir, subjects, session_labels, target_length, fs=1000):
    all_data = []
    all_labels = []

    # Iterate through sessions (1, 2, 3)
    for session_idx in range(3):
        session_folder = os.path.join(data_dir, str(session_idx + 1))  # Session folders named 1, 2, 3

        # Iterate through subjects
        for subject in subjects:
            # Find the .mat file for the subject in the session folder
            # session_files = [f for f in os.listdir(session_folder) if
            #                 f.startswith(f'{subject}_') and f.endswith('.mat')]
            # if len(session_files) != 1:
            #    raise ValueError(f"Subject {subject} has multiple or no session files in {session_folder}")

            session_files = [f for f in os.listdir(session_folder)]
            session_file = session_files[0]
            session_data = loadmat(os.path.join(session_folder, session_file))

            # Extract EEG signals
            eeg_signals = {key: session_data[key] for key in session_data if '_eeg' in key}
            renamed_signals = {f'eeg_{i + 1}': signal for i, (key, signal) in enumerate(eeg_signals.items())}
            sorted_signals = [renamed_signals[f'eeg_{i + 1}'] for i in range(24)]  # 24 trials per session

            # Apply bandpass filter and padding/truncating to each trial
            filtered_trials = [bandpass_filter(trial, 0.5, 50, fs) for trial in sorted_signals]
            padded_trials = [pad_or_truncate_trial(trial, target_length) for trial in filtered_trials]

            # Add data and corresponding labels for this subject and session
            all_data.extend(padded_trials)
            all_labels.extend(session_labels[session_idx])  # Use session_idx to fetch correct labels

    return np.array(all_data), np.array(all_labels)


def main():
    # Load EEG data and preprocess
    data_directory = 'data/eeg/eeg_raw_data'
    subjects = range(1, 16)  # 15 subjects (1 to 15)
    target_length = 64  # Max samples based on the new info for SEED-IV
    eeg_data, labels = load_and_preprocess_eeg_data(data_directory, subjects, get_session_labels(), target_length)
    print(f'EEG Data Shape: {eeg_data.shape}')  # (Total trials, channels, time points)
    print(f'Labels Shape: {labels.shape}')  # (Total trials,)


if __name__ == '__main__':
    main()
