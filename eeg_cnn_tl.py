import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from eeg_mat_load import load_n_merge_raw_data, get_fixed_params
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

pre_proc_loc = 'data/pre-processed/eeg/tl_spec/'
pre_proc_file = 'all_features_s1_new.npy'


# spectrogram_height = 229
# spectrogram_width = 229
#
# frame_size = ((spectrogram_height - 1) * 2)
# sample_size = ((spectrogram_width - 1) * hop_size)


def re_process_data():
    """
      Re-process the raw EEG data into translated data for training.
      :return:
    """

    start_time = time.time()

    eeg_data = load_n_merge_raw_data('data/eeg/eeg_raw_data/1')
    print(f"eeg_data.shape: {eeg_data.shape}")
    fs = 200  # from paper
    all_features = extract_features_from_all_channels(eeg_data, fs)
    np.save(f'{pre_proc_loc}{pre_proc_file}', all_features)  # Save as .npy file

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


def plot_spectogram(t, f, Sxx, y_label, x_label, title):
    """
     This method plots the spectrogram of the EEG data.
     Parameters
     ----------
     t : ndarray
        Array of segment times.
     f : ndarray
        Array of sample frequencies.
     Sxx : ndarray
         Spectrogram of x. By default, the last axis of Sxx corresponds
         to the segment times.
     y_label : The y-axis label
     x_label : The x-axis label
     title :   The title of the plot
    """
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


def plot_for_bands(normalized_band_powers, channel, band):
    """
        Plot band spectrogram for a given channel and frequency band.
        Parameters
        ----------
        normalized_band_powers : ndarray  Array of normalized power spectrograms.
        channel : int The channel number to plot the spectrogram for.
        band : int The band number to plot the spectrogram for.
    """
    power = normalized_band_powers[channel][band]
    plt.plot(power)
    plt.title(f'Spectrogram for {channel} - {band}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Power')
    plt.show()


def generate_spectrogram(eeg_signal, fs, frame_size=456, hop_size=3) -> np.ndarray:
    """
    Generate a spectrogram from the EEG signal using the specified frame size and hop size.

    Parameters
    ----------
    eeg_signal: 1D array of the EEG signal
    fs: Sampling frequency (Hz)
    frame_size: Number of samples per frame
    hop_size: Step size between frames (overlap)

    Returns
    -------
    Sxx: spectrogram array
    """
    f, t, Sxx = spectrogram(eeg_signal, fs, nperseg=frame_size, noverlap=hop_size)
    # plotSpectogram(t,f,Sxx, 'Frequency [Hz]', 'Time [sec]', 'EEG Spectrogram' )
    # sample_size = (229 - 1) * hop_size
    return Sxx


def extract_frequency_band_power(eeg_data, fs):
    """
    Extract power in specific EEG frequency bands (delta, theta, alpha, beta, gamma) from the spectrogram
    for all channels in the eeg_data array.

    Parameters:
    eeg_data: 2D array (channels x time) of the EEG data
    fs: Sampling frequency (Hz)

    Returns:
    dict: Power values for each frequency band for each channel
    """
    # Define frequency bands in Hz
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)  # Adjust based on your requirements
    }

    # Initialize dictionary to store power for each band and each channel
    channel_band_power = {f'eeg_{i + 1}': {} for i in range(eeg_data.shape[0])}

    # Iterate over each channel and calculate spectrogram and band power
    for i, channel_data in enumerate(eeg_data):
        f, t, Sxx = spectrogram(channel_data, fs)

        for band, (low_freq, high_freq) in freq_bands.items():
            idx_band = np.where((f >= low_freq) & (f <= high_freq))[0]
            # Mean power over time for this band
            channel_band_power[f'eeg_{i + 1}'][band] = np.mean(Sxx[idx_band, :], axis=0)

    return channel_band_power


def normalize_band_powers(band_powers):
    """
        Normalize band powers (0-1) using MinMaxScaler.
        Also, band passed signals are further normalized to values
        between 0 and 1 to compute effective spectrograms.
        Then these three spectral signals are converted to three spectrograms
          which is then passed to a pre-trained image classification model to extract transfer learning features.
          Pre-trained model used here is Inception V3
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_band_powers = {}

    for channel, powers in band_powers.items():
        normalized_band_powers[channel] = {}
        for band, power in powers.items():
            # Reshape and normalize between 0 and 1
            power_reshaped = power.reshape(-1, 1)
            normalized_power = scaler.fit_transform(power_reshaped).flatten()
            normalized_band_powers[channel][band] = normalized_power

    return normalized_band_powers


def prepare_spectrogram_for_inception(spec):
    """
        Prepares a spectrogram for InceptionV3 by resizing it to 299x299
        and replicating across 3 channels.
    """
    # Normalize spectrogram
    spec_normalized = np.interp(spec, (spec.min(), spec.max()), (0, 255)).astype(np.uint8)

    # Convert spectrogram to an image (PIL)
    img = Image.fromarray(spec_normalized)

    # Resize to 299x299 (input size for InceptionV3)
    img_resized = img.resize((299, 299))

    # Convert to 3-channel RGB image by replicating the grayscale
    img_rgb = np.stack([img_resized] * 3, axis=-1)

    # Preprocess the image for InceptionV3 (scaling, etc.)
    img_rgb = preprocess_input(img_rgb)

    return img_rgb


def extract_inception_features(img_rgb):
    """
        Extract deep features from the InceptionV3 model.
        :parameter
        :img_rgb:
    """
    # Load pre-trained InceptionV3 model (without top layers)
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Extract features
    img_rgb_expanded = np.expand_dims(img_rgb, axis=0)  # Add batch dimension
    features = model.predict(img_rgb_expanded)

    return features


def process_channel(channel_data, fs):
    """
     Process channel data and return processed data.
     Generate spectrogram and extract InceptionV3 features
    :param channel_data:
    :param fs:
    :return:
    """
    spec = generate_spectrogram(channel_data, fs, 456, 6)
    img_rgb = prepare_spectrogram_for_inception(spec)
    with tf.device('/GPU:0'):
        features = extract_inception_features(img_rgb)
    return features


def extract_features_from_all_channels(eeg_data, fs) -> np.ndarray:
    """
      Extract and merge features from all channels of EEG signal.
    :param eeg_data: the eeg_data as an array of 15x24x62x[y] dimensions
    :param fs: the sample frequency, here 200 for Seed 4
    :return:
      all_features as a nd_array of the same size as eeg_data
    """

    all_features = []
    for subject in range(eeg_data.shape[0]):  # Loop over subjects
        subject_features = []

        for trial in range(eeg_data.shape[1]):  # Loop over trials
            trial_features = []
            # Extract EEG data for this subject, trial, and channel
            trial_data = eeg_data[subject, trial, :, :]

            trial_features = Parallel(n_jobs=-1)(delayed(process_channel)(trial_data[channel], fs)
                                                 for channel in range(eeg_data.shape[2]))
            # trial_features.append(features)

            subject_features.append(trial_features)

        all_features.append(subject_features)

    all_features = np.array(all_features)
    print(f'{all_features.shape}')
    reshaped_features = all_features.reshape(24, 62, 8, 8, 2048)

    return reshaped_features


def build_cnn_model(input_shape, num_classes) -> Sequential:
    """
        Build a CNN model for emotion classification.
        Based on paper:
          https://www.itm-conferences.org/articles/itmconf/pdf/2023/03/itmconf_icdsia2023_02011.pdf
        title: EEG-based Emotion Recognition
          using Transfer Learning Based Feature Extraction and Convolutional Neural Network

        Parameters
        ----------
        input_shape: Shape of the input features (e.g., (8, 8, 2048))
        num_classes: Number of output classes (e.g., 4 for emotions)

        Returns
        ------
        model : Sequential Compiled CNN model
    """
    model = Sequential()

    # 1-4 Convolutional layer
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(lr=0.001, decay=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluate(features, labels, label_types) -> tuple[float, float]:
    """
        Evaluate model on given features and labels.

        Parameters
        ----------
        features: 2D array (channels x time) of the EEG data
                  IncentiveV3 feature list translated from eeg raw data
        labels:  list of session labels provided by seed 4 sample
        label_types: list of labels types provided by seed 4 sample
                     Those are neutral, sad, fear, happy

        :return:
    """
    label_size = len(label_types)
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    print(f'features.shape {features.shape}, labels.shape {labels.shape}')  # Should match the number of labels

    # Split the features and labels into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # x_train = x_train.reshape(-1, 8, 8, 2048)
    # x_test = x_test.reshape(-1, 8, 8, 2048)

    # Convert labels to categorical format (for 4 emotion classes: neutral, sad, fear, happy)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=label_size)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=label_size)

    # Build the CNN model based on the earlier defined architecture
    input_shape = x_train.shape[1:]  # Should be (8, 8, 2048)
    cnn_model = build_cnn_model(input_shape, num_classes=label_size)

    # Train the model
    history = cnn_model.fit(x_train, y_train_cat,
                            epochs=150, batch_size=64,
                            validation_data=(x_test, y_test_cat),
                            callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_acc = cnn_model.evaluate(x_test, y_test_cat)
    print(f"Test accuracy: {test_acc}")
    return test_acc, test_loss


def display_evaluate_model(all_features):
    # Example usage:
    input_shape = (8, 9, 2048)  # Shape of InceptionV3 features
    # Number of emotions (neutral, sad, fear, happy)
    label_types = ['neutral', 'sad', 'fear', 'happy']
    smooth_dirs, labels, eeg_keys = get_fixed_params()

    # cnn_model = build_cnn_model(input_shape, len(label_types))
    # cnn_model.summary()  # Show model architecture
    evaluate(all_features, np.array(labels[0]), label_types)


def main():
    """
    Main method of the program.
    Used for generating and plotting spectrogram for EEG data
    Implementation with SEED-IV
    :return:
    """

    re_process_data()
    all_features = np.load(f'{pre_proc_loc}{pre_proc_file}')
    print(f"all_features.shape: {all_features.shape}, index 1 shape {all_features[0].shape}")
    reshaped_features = np.squeeze(all_features)  # Removes the redundant dimension
    # Should be (62, 8, 8, 2048)
    print(f"reshaped_features.shape: {reshaped_features.shape}")
    # reshaped_features = reshaped_features.reshape(24, -1)
    # display_evaluate_model(reshaped_features)


def main_sample():
    """
    Main method for generating and plotting spectrogram for EEG data.
    Sample code used for testing each step
    :return:
    """
    # spec = generate_spectrogram(eeg_data[0, :], fs)  # Use one EEG channel
    # print(spec.shape)

    # channel_band_powers = extract_frequency_band_power(eeg_data, fs)
    # Output the band powers for each channel
    # for channel, bands in channel_band_powers.items():
    #     for band, power in bands.items():
    #         print(f"{channel} - {band} band power: {np.mean(power)}")

    # normalized_band_powers = normalize_band_powers(channel_band_powers)
    # plot_for_bands(normalized_band_powers, 'eeg_1', 'delta')
    # plot_for_bands(normalized_band_powers, 'eeg_1', 'theta')
    # plot_for_bands(normalized_band_powers, 'eeg_1', 'alpha')
    # plot_for_bands(normalized_band_powers, 'eeg_1', 'beta')
    # plot_for_bands(normalized_band_powers, 'eeg_1', 'gamma')

    # img_rgb = prepare_spectrogram_for_inception(spec)
    # features = extract_inception_features(img_rgb)
    # print(f"Extracted feature shape: {features.shape}")


if __name__ == '__main__':
    main()
