import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt

from cnn.eeg_mat_load import get_session_labels

# Bi HDM

# Bandpass filter design

pre_proc_loc = '../data/pre-processed/eeg/bi_hdm/'
pre_proc_data_file = 'feature_data.npy'
pre_proc_label_file = 'feature_label.npy'
pre_proc_sub_file = 'feature_subject_array.npy'


def load_pre_processed_data():
    """
     Use this method to load data from default locations
    :return:
    """
    data_f_nm = pre_proc_loc + pre_proc_data_file
    label_f_nm = pre_proc_loc + pre_proc_label_file
    subject_f_nm = pre_proc_loc + pre_proc_sub_file
    return load_processed_data(data_f_nm, label_f_nm, subject_f_nm)


def load_processed_data(data_file_name: str,
                        label_file_name: str,
                        sub_file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Load processed EEG data for LOSO cross-validation
     Use this method to load data from custom locations
    :param data_file_name: str, file name for eeg_data to be loaded as numpy array
    :param label_file_name: str, file name for label_data to be loaded as numpy array
    :param sub_file_name: str, file name for sub_array_data to be loaded as numpy array
    :return:
    """
    eeg_data = np.load(f'{data_file_name}')
    labels = np.load(f'{label_file_name}')
    sub_array = np.load(f'{sub_file_name}')
    return eeg_data, labels, sub_array


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


def load_and_grp_eeg_by_sub(data_dir, subjects, session_labels, target_length, fs=1000):
    """
    Loads, preprocesses, and groups EEG data by subject, merging data from multiple sessions.

    Parameters
    ----------
    data_dir : str
        The base directory containing EEG data folders.

    subjects : list,
        List of subject identifiers.

    session_labels : list of list,
        List of session labels for each subject, where each entry corresponds to the labels for a session.

    target_length : int
        The target length to which each EEG trial should be padded or truncated.

    fs : int
        The sampling frequency of the EEG data.

    Returns
    -------
    grouped_data : dict
        A dictionary where the key is the subject ID and the value is the subject's EEG data.

    grouped_labels : dict
        A dictionary where the key is the subject ID and the value is the corresponding labels.
    """
    grouped_data = {}
    grouped_labels = {}
    plotted = False

    # Iterate through sessions (1, 2, 3)
    for session_idx in range(3):
        session_folder = os.path.join(data_dir, str(session_idx + 1))  # Session folders named 1, 2, 3

        # Iterate through subjects
        for subject in subjects:

            # Match files for the current subject
            subject_files = [f for f in os.listdir(session_folder) if f.startswith(f'{subject}_')]
            if len(subject_files) != 1:
                raise ValueError(f"Ambiguous or missing files for subject {subject} in session {session_idx + 1}")
            session_file = subject_files[0]

            # Find the .mat file for the subject in the session folder
            try:
                session_data = loadmat(os.path.join(session_folder, session_file))
            except Exception as e:
                print(f"Error loading file for subject {subject}, session {session_idx + 1}: {e}")
                continue

            # subject_files = [f for f in os.listdir(session_folder)]
            # session_file = subject_files[0]
            # session_data = loadmat(os.path.join(session_folder, session_file))

            # Extract EEG signals
            eeg_signals = {key: session_data[key] for key in session_data if '_eeg' in key}
            renamed_signals = {f'eeg_{i + 1}': signal for i, (key, signal) in enumerate(eeg_signals.items())}
            sorted_signals = [renamed_signals[f'eeg_{i + 1}'] for i in range(24)]  # 24 trials per session

            # Apply bandpass filter and padding/truncating to each trial
            filtered_trials = [bandpass_filter(trial, 0.5, 50, fs) for trial in sorted_signals]
            padded_trials = [pad_or_truncate_trial(trial, target_length) for trial in filtered_trials]

            # Plot the filtered trials for filter/padding and domain verification
            if plotted is False:
                plot_filter_padded_data(filtered_trials, "Filtered Trial")
                plot_filter_padded_data(padded_trials, "Padded Trial")
                plot_frq_domain_verification(filtered_trials)
                plotted = True

            # Check for NaNs in filtered data
            if any(np.isnan(trial).any() for trial in filtered_trials):
                print(f"NaN values detected in filtered data for subject {subject}, session {session_idx + 1}")
                continue

            # If the subject has not been added yet, initialize their data and label lists
            if subject not in grouped_data:
                grouped_data[subject] = []
                grouped_labels[subject] = []

            # Append the trials and corresponding labels for this subject and session
            grouped_data[subject].extend(padded_trials)
            grouped_labels[subject].extend(session_labels[session_idx])  # Use session_idx to fetch correct labels

    # Convert lists to numpy arrays for easier manipulation
    for subject in grouped_data:
        grouped_data[subject] = np.array(grouped_data[subject])
        grouped_labels[subject] = np.array(grouped_labels[subject])

    return grouped_data, grouped_labels


def plot_filter_padded_data(trials: list, trial_title: str):
    """
     This method is used to plot the trail data filtered and padded EEG data.
    :param trials: the number of trials
    :return: None
    """
    plt.plot(trials[0][0, :])  # First channel of the first trial
    plt.title(trial_title)
    plt.xlabel("Time Points")
    plt.ylabel("Amplitude")
    plt.show()


def plot_frq_domain_verification(trials):
    """
     This method is used to plot the data to verify the band domains
    :param trials: the number of trials
    :return: None
    """
    freqs, psd = welch(trials[0][0, :], fs=1000)
    plt.semilogy(freqs, psd)
    plt.title("Power Spectral Density of Filtered Trial")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, 60)
    plt.show()


def main():
    """
     Preprocess EEG data for LOSO cross-validation
     Then save the eeg_data, labels and subject array
    :param data_file_name: str, file name for eeg_data to be saved as numpy array
    :param label_file_name: str, file name for label_data to be saved as numpy array
    :param sub_file_name: str, file name for sub_array_data to be saved as numpy array
    :return: None
    """
    # Load EEG data and preprocess
    data_directory = '../data/eeg/eeg_raw_data'
    subjects = range(1, 16)  # 15 subjects (1 to 15)
    target_length = 64  # Max samples based on the new info for SEED-IV
    grp_data, grp_labels = load_and_grp_eeg_by_sub(data_directory, subjects,
                                                   get_session_labels(),
                                                   target_length)

    # Prepare data for LOSO cross-validation
    eeg_data = np.concatenate([grp_data[subject] for subject in subjects], axis=0)
    labels = np.concatenate([grp_labels[subject] for subject in subjects], axis=0)
    print(f'EEG Data Shape: {eeg_data.shape}')  # (Total trials, channels, time points)
    print(f'Labels Shape: {labels.shape}')  # (Total trials,)

    # Generate subjects array for LOSO
    sub_array = np.concatenate([[sub] * len(grp_data[sub]) for sub in subjects])

    np.save(pre_proc_data_file, eeg_data)  # Save as .npy file
    np.save(pre_proc_label_file, labels)  # Save as .npy file
    np.save(pre_proc_sub_file, sub_array)  # Save as .npy file


if __name__ == '__main__':
    main()
