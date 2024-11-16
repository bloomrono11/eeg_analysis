import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import welch
from scipy.stats import norm, f_oneway

from cnn.eeg_bi_hdm_pre_process import load_pre_processed_data

# Define the EEG frequency bands
# Larger segment for low frequencies
frequency_bands = {
    "Delta (0.5-4 Hz)": {"range": (0.5, 4), "nperseg": 4000},
    "Theta (4-8 Hz)": {"range": (4, 8), "nperseg": 2000},
    "Alpha (8-13 Hz)": {"range": (8, 13), "nperseg": 1000},
    # "Beta (13-30 Hz)": {"range": (13, 30), "nperseg": 1000},
    # "Gamma (30+ Hz)": {"range": (30, 100), "nperseg": 1000}
}


def summary_statistics(eeg_data):
    """
    Basic Characteristics and Summary Statistics
    # Assuming eeg_data is a NumPy array with dimensions [n_trials, n_channels, n_timepoints]
    :param eeg_data:
    :return:
    """
    n_trials, n_channels, n_timepoints = eeg_data.shape
    print("Number of trials:", n_trials)
    print("Number of channels:", n_channels)
    print("Number of time points per trial:", n_timepoints)

    # Summary statistics for each channel
    mean_per_channel = np.mean(eeg_data, axis=(0, 2))
    variance_per_channel = np.var(eeg_data, axis=(0, 2))
    std_per_channel = np.std(eeg_data, axis=(0, 2))

    print("Mean per channel:", mean_per_channel)
    print("Variance per channel:", variance_per_channel)
    print("Standard deviation per channel:", std_per_channel)
    # Plot the mean and std per channel
    plot_gaussian_distribution(mean_per_channel, std_per_channel)
    plot_mean_std(mean_per_channel, std_per_channel)


def plot_mean_std(mean_per_channel, std_per_channel):
    """
    Visualize the distribution of the mean and standard deviation per channel using histograms and KDE.
    """
    plt.figure(figsize=(14, 6))

    # Plot histogram and KDE for the mean per channel
    plt.subplot(1, 2, 1)
    sns.histplot(mean_per_channel, kde=True, color="blue", bins=15)
    plt.title("Distribution of Mean per Channel")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")

    # Plot histogram and KDE for the standard deviation per channel
    plt.subplot(1, 2, 2)
    sns.histplot(std_per_channel, kde=True, color="green", bins=15)
    plt.title("Distribution of Standard Deviation per Channel")
    plt.xlabel("Standard Deviation Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def calculate_band_power1(eeg_data, fs=1000, band_info=None):
    """
    Calculate the average power for a specific frequency band across trials for each channel.

    :param eeg_data: np.ndarray, EEG data of shape [n_trials, n_channels, n_timepoints]
    :param fs: int, Sampling frequency in Hz
    :param band_info: dict, Contains 'range' (frequency range) and 'nperseg' (segment length)
    :return: list of average power values for each channel
    """
    if not band_info:
        raise ValueError("Band information must be provided.")

    band_range = band_info["range"]
    nperseg = band_info["nperseg"]

    band_powers = []
    for ch in range(eeg_data.shape[1]):
        power_per_trial = []
        for trial in range(eeg_data.shape[0]):
            freqs, psd = welch(eeg_data[trial, ch, :], fs=fs, nperseg=nperseg)
            band_idx = np.logical_and(freqs >= band_range[0], freqs < band_range[1])
            if np.any(band_idx):
                power_per_trial.append(np.mean(psd[band_idx]))
            else:
                power_per_trial.append(0)  # Assign 0 if no valid frequencies
        band_powers.append(power_per_trial)
    return band_powers


def calculate_band_power(eeg_data, fs=1000, band_info=None) -> list:
    """
    Dynamically adjusts nperseg to avoid exceeding the trial length.
    """
    if not band_info:
        raise ValueError("Band information must be provided.")

    band_range = band_info["range"]
    nperseg = min(band_info["nperseg"], eeg_data.shape[2])  # Ensure nperseg <= n_timepoints

    band_power = []
    for ch in range(eeg_data.shape[1]):
        power_per_trial = []
        for trial in range(eeg_data.shape[0]):
            freqs, psd = welch(eeg_data[trial, ch, :], fs=fs, nperseg=nperseg)
            band_idx = np.logical_and(freqs >= band_range[0], freqs < band_range[1])
            if np.any(band_idx):
                power_per_trial.append(np.mean(psd[band_idx]))
            else:
                power_per_trial.append(0)  # Handle cases with no valid frequencies
        band_power.append(power_per_trial)
    return band_power


def check_constant_channels(band_power):
    """
    Identifies channels with constant power values.
    """
    constant_channels = [ch for ch, powers in enumerate(band_power) if np.all(np.isclose(powers, powers[0]))]
    if constant_channels:
        print(f"Warning: Channels with constant power detected: {constant_channels}")
    return constant_channels


def perform_anova(band_power):
    """
    Perform one-way ANOVA across the channels.

    :param band_power: list of lists, where each inner list contains band power values for a specific channel
    """
    # Unpack the list of lists into arguments for f_oneway
    anova_result = f_oneway(*band_power)
    print("ANOVA test results")
    print("F-statistic:", anova_result.statistic)
    print("p-value:", anova_result.pvalue)


def plot_gaussian_distribution(mean_per_channel, std_per_channel):
    """
    Normalizes and plots mean and standard deviation distributions as a single normalized bell curve.
    """
    # Normalize the mean and std values to fit a standard normal distribution
    norm_mean = (mean_per_channel - np.mean(mean_per_channel)) / np.std(mean_per_channel)
    norm_std = (std_per_channel - np.mean(std_per_channel)) / np.std(std_per_channel)

    # Plot the normalized distributions
    x_values = np.linspace(-3, 3, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, norm.pdf(x_values), color='black', linestyle='--', label='Standard Normal Distribution')

    sns.kdeplot(norm_mean, color="blue", label="Normalized Mean Distribution")
    sns.kdeplot(norm_std, color="green", label="Normalized Std Distribution")

    plt.title("Combined Normalized Distribution of Mean and Standard Deviation per Channel")
    plt.xlabel("Normalized Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def power_spectrum(eeg_data, n_channels: int):
    # Power Spectral Density (PSD) for a trial and each channel
    fs = 1000  # Sampling frequency in Hz
    trial_idx = 0  # Select a trial

    plt.figure(figsize=(15, 15))
    for ch in range(n_channels):
        freqs, psd = welch(eeg_data[trial_idx, ch, :], fs, nperseg=256)
        plt.plot(freqs, psd, label=f'Channel {ch + 1}')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.title("Power Spectral Density for a Single Trial")
    plt.legend(loc="upper right")
    plt.show()


def power_spectrum_avg(eeg_data, n_channels: int,
                       fs: int = 1000,
                       trial_idx: int = 0,
                       average_trials: bool = False):
    """
    Compute and plot Power Spectral Density (PSD) for each channel in a trial, or averaged across trials.

    :param eeg_data: np.ndarray, EEG data of shape [n_trials, n_channels, n_timepoints]
    :param n_channels: int, number of channels in the EEG data
    :param fs: int, sampling frequency in Hz
    :param trial_idx: int, index of the trial to visualize; ignored if average_trials is True
    :param average_trials: bool, whether to average PSD across trials
    """
    plt.figure(figsize=(15, 15))

    if average_trials:
        # Compute the average PSD across all trials for each channel
        avg_psd = []
        for ch in range(n_channels):
            psds = [welch(eeg_data[trial, ch, :], fs, nperseg=256)[1] for trial in range(eeg_data.shape[0])]
            avg_psd.append(np.mean(psds, axis=0))
        freq = welch(eeg_data[0, 0, :], fs, nperseg=256)[0]

        # Plot the averaged PSD
        for ch in range(n_channels):
            plt.plot(freq, avg_psd[ch], label=f'Channel {ch + 1}')
    else:
        # Compute PSD for a single trial
        for ch in range(n_channels):
            freq, psd = welch(eeg_data[trial_idx, ch, :], fs, nperseg=256)
            plt.plot(freq, psd, label=f'Channel {ch + 1}')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.title(
        "Power Spectral Density" + (" (Averaged Across Trials)" if average_trials else f" for Trial {trial_idx + 1}"))
    plt.legend(loc="upper right")  # ,  bbox_to_anchor=(0.75, -0.075), ncol=3, fontsize=12)
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.show()


def band_powers(eeg_data, fs=1000, nperseg=256):
    """
    Calculates the average power for each EEG frequency band across trials and channels.

    :param eeg_data: np.ndarray, EEG data of shape [n_trials, n_channels, n_timepoints]
    :param fs: int, Sampling frequency in Hz
    :param nperseg: int, Length of each segment for Welch method
    :return: dict containing average band power for each channel across trials
    """
    # Define frequency bands
    bands = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30+ Hz)": (30, fs / 2)
    }

    # Initialize a dictionary to hold power values
    band_pow = {band: [] for band in bands}

    # Iterate over each channel
    for ch in range(eeg_data.shape[1]):
        power_per_band = {band: [] for band in bands}

        # Calculate PSD across all trials for a channel
        for trial in range(eeg_data.shape[0]):
            freq, psd = welch(eeg_data[trial, ch, :], fs=fs, nperseg=nperseg)

            # For each frequency band, calculate mean power within the band range
            for band, (low, high) in bands.items():
                band_idx = np.logical_and(freq >= low, freq < high)
                power_per_band[band].append(np.mean(psd[band_idx]))

        # Average across trials for each band
        for band in bands:
            band_pow[band].append(np.mean(power_per_band[band]))

    return band_pow


def plot_band_power(band_pow, fs=1000):
    """
    Plots the average power of each EEG band across channels.

    :param band_pow: dict, average power values for each band and channel
    :param fs: the sampling frequency in Hz
    """
    plt.figure(figsize=(12, 8))

    # For each band, plot the power values across channels
    for band, power_values in band_pow.items():
        plt.plot(power_values, label=band)

    plt.xlabel("Channel")
    plt.ylabel("Average Power (V^2/Hz)")
    plt.title(f"Average Power in EEG Frequency {fs} Bands Across Channels")
    plt.legend()
    plt.show()


def correlation_per_channel(eeg_data):
    # Calculate mean per channel over trials and time points
    channel_means = eeg_data.mean(axis=2)  # Average across time points
    correlation_matrix = np.corrcoef(channel_means.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title("Channel-Wise Correlation")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.show()


def hemisphere_analysis(eeg_data, n_channels: int):
    # Assuming left hemisphere channels are first half and right are second half
    left_channels = eeg_data[:, :n_channels // 2, :]
    right_channels = eeg_data[:, n_channels // 2:, :]

    # Calculate mean and variance per hemisphere
    mean_left = left_channels.mean(axis=(0, 2))
    variance_left = left_channels.var(axis=(0, 2))

    mean_right = right_channels.mean(axis=(0, 2))
    variance_right = right_channels.var(axis=(0, 2))

    # Plot comparison of mean and variance for left vs. right hemisphere
    plt.figure(figsize=(10, 6))
    plt.plot(mean_left, label="Left Hemisphere Mean")
    plt.plot(mean_right, label="Right Hemisphere Mean")
    plt.xlabel("Channel Index")
    plt.ylabel("Mean Amplitude")
    plt.title("Left vs. Right Hemisphere Mean Amplitude")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(variance_left, label="Left Hemisphere Variance")
    plt.plot(variance_right, label="Right Hemisphere Variance")
    plt.xlabel("Channel Index")
    plt.ylabel("Variance")
    plt.title("Left vs. Right Hemisphere Variance")
    plt.legend()
    plt.show()

    # Inter-hemispheric asymmetry: absolute difference and ratio
    asymmetry_diff = np.abs(left_channels - right_channels)
    asymmetry_ratio = left_channels / (right_channels + 1e-5)

    # Plot asymmetry for a single channel pair (e.g., channel 0 from each hemisphere)
    channel_pair_idx = 0
    plt.figure(figsize=(10, 6))
    plt.plot(asymmetry_diff[:, channel_pair_idx, :].mean(axis=0), label="Mean Absolute Difference")
    plt.plot(asymmetry_ratio[:, channel_pair_idx, :].mean(axis=0), label="Mean Ratio")
    plt.xlabel("Time Points")
    plt.ylabel("Asymmetry")
    plt.title("Inter-Hemispheric Asymmetry for a Channel Pair")
    plt.legend()
    plt.show()


def main():
    eeg_data, labels, subject_arr = load_pre_processed_data()

    print(f'EEG Data Shape: {eeg_data.shape}')  # (Total trials, channels, time points)
    print(f'Labels Shape: {labels.shape}')  # (Total trials,)
    print(f'Subject Array Shape: {subject_arr.shape}')  # (Total Subjects)
    # summary_statistics(eeg_data)
    # power_spectrum(eeg_data, n_channels=62)
    power_spectrum_avg(eeg_data, 62, average_trials=True)

    #band_power_values = band_powers(eeg_data)
    #plot_band_power(band_power_values, fs=1000)

    # Down sampled
    #band_power_values = band_powers(eeg_data, fs=200)
    #plot_band_power(band_power_values, fs=200)

    #correlation_per_channel(eeg_data)
    #hemisphere_analysis(eeg_data, n_channels=62)

    # Anova Test
    # Iterate through each band, calculate power, and perform ANOVA
    # for band_name, band_info in frequency_bands.items():
    #     print(f"\nPerforming ANOVA for {band_name}")
    #
    #     # Calculate band power for the current band
    #     band_power = calculate_band_power(eeg_data, fs=1000, band_info=band_info)
    #
    #     # Check for constant power channels
    #     constant_channels = check_constant_channels(band_power)
    #     if constant_channels:
    #         print(f"Skipping ANOVA for {band_name} due to constant channels.")
    #         continue
    #
    #     # Perform ANOVA test across channels
    #     perform_anova(band_power)


if __name__ == '__main__':
    main()
