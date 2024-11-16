import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from eeg_bi_hdm_pre_process import load_and_grp_eeg_by_sub, load_processed_data
from cnn.eeg_mat_load import get_session_labels


class GradientReversalLayer(layers.Layer):
    def __init__(self, lambda_):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    @tf.custom_gradient
    def call(self, inputs):
        def grad(dy):
            return -self.lambda_ * dy  # Reverses the gradient

        return inputs, grad


def pair_hemispheres(left_lstm, right_lstm):
    """
       Perform pairwise operations between left and right hemispheres.

       Parameters
       ---------
       left_lstm : tensor
           LSTM output from the left hemisphere.

       right_lstm : tensor
           LSTM output from the right hemisphere.

       Returns
       ------
       Concatenated tensor of pairwise operations (subtraction, division, multiplication).
       """

    def pairwise_operation(left, right):
        """
            Pairwise operations between left and right hemispheres
            :param left:
            :param right:
            :return:
        """
        return tf.keras.layers.Concatenate()([
            tf.abs(left - right),
            left / (right + 1e-5),
            left * right
        ])

    return pairwise_operation(left_lstm, right_lstm)


def create_bi_hdm_model_as_paper(input_shape, lambda_val=1.0) -> models.Model:
    """
        Creates the bi_hdm model originally discussed in the paper
        input_shape = ( 62/2, 64) as per paper
        Parameters
        ---------
        input_shape: The nd array of shape (31, 64)
        lambda_val: float The gradient reversal strength.

        Returns
        ------
        model: keras.Model
        The BiHDM model with adversarial domain adaptation.
    """
    # Left hemisphere RNN (horizontal and vertical)
    input_left = layers.Input(shape=input_shape)
    left_lstm_h = layers.LSTM(32, return_sequences=False)(input_left)
    left_lstm_v = layers.LSTM(32, return_sequences=False)(input_left)

    # Right hemisphere RNN (horizontal and vertical)
    input_right = layers.Input(shape=input_shape)
    right_lstm_h = layers.LSTM(32, return_sequences=False)(input_right)
    right_lstm_v = layers.LSTM(32, return_sequences=False)(input_right)

    # Concatenate pairwise results
    combined_h = pair_hemispheres(left_lstm_h, right_lstm_h)
    combined_v = pair_hemispheres(left_lstm_v, right_lstm_v)

    # Reshape the combined outputs to add the time step dimension (1 time step here)
    combined_h_reshaped = layers.Reshape((1, combined_h.shape[-1]))(combined_h)
    combined_v_reshaped = layers.Reshape((1, combined_v.shape[-1]))(combined_v)

    # Higher-level discrepancy feature extraction
    combined_h_lstm = layers.LSTM(32)(combined_h_reshaped)
    combined_v_lstm = layers.LSTM(32)(combined_v_reshaped)

    # Final concatenation of horizontal and vertical
    combined_final = layers.Concatenate()([combined_h_lstm, combined_v_lstm])

    # Final classification layer, 4 emotions: neutral, sad, fear, happy
    emotion_output = layers.Dense(4, activation='softmax', name="emotion_output")(combined_final)

    # Gradient Reversal Layer for adversarial training
    grl = GradientReversalLayer(lambda_=lambda_val)(combined_final)

    # Domain Discriminator (simple dense layers for domain classification)
    domain_fc = layers.Dense(64, activation='relu')(grl)
    domain_output = layers.Dense(1, activation='sigmoid', name="domain_output")(domain_fc)

    model = models.Model(inputs=[input_left, input_right], outputs=[emotion_output, domain_output])

    # Define a learning rate schedule instead of using `decay`
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.003,
        decay_steps=100000,
        decay_rate=0.95,
        staircase=True
    )

    # optimizer = SGD(learning_rate=0.003, momentum=0.9, decay=0.95)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    # model with two losses: emotion classification and domain classification
    model.compile(optimizer=optimizer,
                  loss={
                      "emotion_output": "sparse_categorical_crossentropy",
                      "domain_output": "binary_crossentropy"
                  },
                  metrics={
                      "emotion_output": "accuracy",
                      "domain_output": "accuracy"
                  })
    return model


def create_bi_hdm_lstm(input_shape, units, dropout=0.2, regular=None, first_seq=True, sec_seq=False):
    """
    Creates and define the bi_hdm model lstm section only
    :param input_shape:
    :param units:
    :param dropout:
    :param regular:
    :param sec_seq:
    :param first_seq:
    :return:
    """

    input_left = layers.Input(shape=input_shape)
    lstm = layers.LSTM(units, return_sequences=first_seq, kernel_regularizer=regular)(input_left)
    lstm = layers.BatchNormalization()(lstm)
    lstm = layers.Dropout(dropout)(lstm)
    lstm = layers.LSTM(units, return_sequences=sec_seq)(lstm)
    lstm = layers.BatchNormalization()(lstm)
    lstm = layers.Dropout(dropout)(lstm)
    return lstm, input_left


def bi_hdm_complete_model(temp_out, input_left, input_right, lr, regular=None, weight_decay=None):
    """
    Completes the bi hdm model by mapping output to input, adding optimizer, learning rate and regularization weights
    :param temp_out:
    :param input_left:
    :param input_right:
    :param lr:
    :param regular:
    :param weight_decay:
    :return:
    """
    temp_out = tf.cast(temp_out, tf.float32)
    output = layers.Dense(4, activation='softmax', kernel_regularizer=regular)(temp_out)

    model = models.Model(inputs=[input_left, input_right], outputs=output)

    # Compile the model with a higher learning rate
    optimizer = optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_bi_hdm_model_improved(input_shape):
    # Left hemisphere LSTM
    regular = regularizers.l2(0.01)
    left_lstm, input_left = create_bi_hdm_lstm(input_shape, 64, 0.3, regular)

    # Right hemisphere LSTM
    right_lstm, input_right = create_bi_hdm_lstm(input_shape, 64, 0.3, regular)

    paired_features = pair_hemispheres(left_lstm, right_lstm)

    return bi_hdm_complete_model(paired_features, input_left, input_right, 0.0004)


def create_bi_hdm_model_modified_attn(input_shape):
    """
     Modified model for improving the accuracy of the BiHDM model using Attention
    :param input_shape: The shape of the nd array of the input vectors
    :return: the model using Attention and LSTM
    """
    # Left hemisphere LSTM
    left_lstm, input_left = create_bi_hdm_lstm(input_shape, 128, 0.4, None, True, True)

    # Right hemisphere LSTM
    right_lstm, input_right = create_bi_hdm_lstm(input_shape, 128, 0.4, None, True, True)

    # Attention Mechanism
    attention = layers.Attention()([left_lstm, right_lstm])
    attn_out = layers.Flatten()(attention)

    regular = regularizers.l2(0.01)
    return bi_hdm_complete_model(attn_out, input_left, input_right, 0.001, regular, 1e-4)


def create_bi_hdm_model_modified_multi_head_att(input_shape):
    """
     Modified model for improving the accuracy of the BiHDM model using Multi Head Attention
    :param input_shape: The shape of the nd array of the input vectors
    :return: the model using Multi Head Attention and LSTM
    """
    # Left hemisphere LSTM
    left_lstm, input_left = create_bi_hdm_lstm(input_shape, 128, 0.5, None, True, True)

    # Right hemisphere LSTM
    right_lstm, input_right = create_bi_hdm_lstm(input_shape, 128, 0.5, None, True, True)

    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
    attention = attention(left_lstm, right_lstm)
    attn_out = layers.GlobalAveragePooling1D()(attention)

    regular = regularizers.l2(0.01)
    return bi_hdm_complete_model(attn_out, input_left, input_right, 0.0005, regular, 1e-5)


def prepare_bi_hdm_input(eeg_data: np.array, target_length: int) -> tuple[list, list]:
    """
        Prepare the input data for BiHDM
        Split data into left and right hemispheres (using 31 paired electrodes)
        Parameters
        ---------
        eeg_data: nd array with shape [n_samples, n_channels, n_time]
        target_length: int the target length used for the eeg data

        Returns
        ------
        left_hemisphere and right_hemisphere as list
    """
    # Prepare the input data for BiHDM
    # Split data into left and right hemispheres (using 31 paired electrodes)
    left_hemisphere = eeg_data[:, :31, :]  # First 31 electrodes for the left hemisphere
    right_hemisphere = eeg_data[:, 31:, :]  # Last 31 electrodes for the right hemisphere

    return left_hemisphere, right_hemisphere


def plot_confusion_matrix_from_cm(conf_matrix):
    """
       Plot the confusion matrix for the class labels defined as per Seed 4
       Parameters
       ---------
       conf_matrix: array with shape [n_samples, n_classes]
    """
    class_names = ['Neutral', 'Sad', 'Fear', 'Happy']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_loss(train_loss, test_loss):
    epochs = range(1, len(train_loss) + 1)  # Ensure x-axis starts at 1

    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.grid(True)
    plt.show()


def evaluate_model_4_paper(eeg_data, labels, target_length, subjects, epochs, input_shape):
    """
    Leave-One-Subject-Out (LOSO) Cross-Validation for BiHDM model.

    Parameters
    ----------
    eeg_data: ndarray
        The EEG data with shape [n_samples, n_channels, n_time].

    labels: ndarray
        The labels corresponding to the EEG data.

    target_length: int
        The target length for the EEG data.

    subjects: ndarray
        An array identifying which subject each data point belongs to.

    epochs: int
        The number of epochs for training.
    input_shape: set a set of fixed value (31, 64)

    Returns
    -------
    Mean and standard deviation of accuracy across subjects.
    """
    logo = LeaveOneGroupOut()  # Leave-One-Subject-Out cross-validator
    accuracies = []
    confusion_matrices = []

    for train_index, test_index in logo.split(eeg_data, labels, groups=subjects):
        # Split the data into training and test sets
        x_train, x_test = eeg_data[train_index], eeg_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Split into left and right hemispheres
        left_train, right_train = prepare_bi_hdm_input(x_train, target_length)
        left_test, right_test = prepare_bi_hdm_input(x_test, target_length)

        # Create and compile the model
        model = create_bi_hdm_model_as_paper(input_shape=input_shape, lambda_val=2)

        # Train the model with early stopping
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit([left_train, right_train], y_train,
                            epochs=epochs, batch_size=200,
                            validation_split=0.2,
                            callbacks=[early_stopping], verbose=0)

        # Make predictions on the test set
        y_pred = model.predict([left_test, right_test])[0]
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Calculate accuracy and confusion matrix for the current subject
        accuracy = accuracy_score(y_test, y_pred_labels)
        accuracies.append(accuracy)
        conf_matrix = confusion_matrix(y_test, y_pred_labels)
        confusion_matrices.append(conf_matrix)
        # Plot the loss using the history object
        plot_loss(history.history['loss'], history.history['val_loss'])
        # print(f"Subject {subjects[test_index][0]} - Accuracy: {accuracy:.4f}")

    # Calculate mean accuracy and standard deviation across subjects
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"LOSO Cross-Validation - Mean Accuracy: {mean_accuracy:.4f}, STD: {std_accuracy:.4f}")

    # Optionally, you can also average confusion matrices for better visualization
    avg_conf_matrix = np.mean(confusion_matrices, axis=0)
    plot_confusion_matrix_from_cm(avg_conf_matrix)
    print(f"Confusion Matrix:\n{avg_conf_matrix}")

    return mean_accuracy, std_accuracy


def evaluate_model(eeg_data, labels, target_length, seed, epochs, model: models.Model):
    """
        Prepare the input data for BiHDM
        Split data into left and right hemispheres (using 31 paired electrodes)
        Parameters
        ---------
        eeg_data: the ndarray with shape [n_samples, n_channels, n_time]
        labels: the y pred values provided by Seed 4 dataset owners
        target_length: 64 int the target length used for the eeg data
        seed: seed for random number generator
        epochs: number of epochs for training
        model: the model to evaluate

        Returns
        ------
        left_hemisphere and right_hemisphere as list

    """
    # eeg_data and labels already preprocessed with target_length=64
    left_data, right_data = prepare_bi_hdm_input(eeg_data, target_length)

    # Adjust weights based on class frequency or performance
    class_weights = {0: 1, 1: 2.0, 2: 2, 3: 1.5}

    # Split the data
    # x_train_l, x_test_l, x_train_r, x_test_r, y_train, y_test = train_test_split(
    #    left_data, right_data, labels, test_size=0.2, random_state=seed)

    x_train_l, x_test_l, y_train, y_test \
        = train_test_split(left_data, labels, test_size=0.2, random_state=seed)
    x_train_r, x_test_r, _, _ \
        = train_test_split(right_data, labels, test_size=0.2, random_state=seed)

    # Train the model between 50 and 100 with early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit([x_train_l, x_train_r], y_train, epochs=epochs, batch_size=200,
                        validation_split=0.2,
                        class_weight=class_weights,
                        callbacks=[early_stopping], verbose=0)

    # Evaluate the model
    y_pred = model.predict([x_test_l, x_test_r])
    y_pred_labels = np.argmax(y_pred[0], axis=1)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred_labels)
    conf_matrix = confusion_matrix(y_test, y_pred_labels)

    training_loss = history.history['loss']  # Training loss
    validation_loss = history.history['val_loss']  # Validation loss

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Train Loss: {training_loss[0:2]}, Test Loss {validation_loss[0:2]}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    plot_confusion_matrix_from_cm(conf_matrix)
    plot_loss(training_loss, validation_loss)


def k_fold_cross_validation(eeg_data, labels, target_length, seed, epochs, n_splits=5):
    """
    Perform k-fold cross-validation with the BiHDM model.
    Parameters:
    ----------
    eeg_data: ndarray with shape [n_samples, n_channels, n_time]
    labels: ndarray with the emotion labels
    target_length: int, the target length used for the EEG data
    seed: int, seed for random number generator
    n_splits: int, number of k-folds for cross-validation

    Returns:
    -------
    Average accuracy and confusion matrix across folds.
    """
    # Preprocess input data into left and right hemispheres
    left_data, right_data = prepare_bi_hdm_input(eeg_data, target_length)
    input_shape = (31, target_length)  # 31 electrodes, 64 time points

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Lists to store results for each fold
    accuracy_scores = []
    confusion_matrices = []

    fold_no = 1

    for train_idx, test_idx in kfold.split(left_data):
        print(f"Training fold {fold_no}...")

        # Split the data based on train/test indices for both hemispheres and labels
        x_train_l, x_test_l = left_data[train_idx], left_data[test_idx]
        x_train_r, x_test_r = right_data[train_idx], right_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Create a new instance of the model for each fold
        model = create_bi_hdm_model_modified_multi_head_att(input_shape)

        # Dynamically compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=y_train)
        class_weights = dict(enumerate(class_weights))

        # Train the model
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit([x_train_l, x_train_r], y_train, epochs=epochs, batch_size=200,
                            validation_split=0.2, class_weight=class_weights, callbacks=[early_stopping])

        # Evaluate the model on the test set for this fold
        y_pred = model.predict([x_test_l, x_test_r])
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Calculate accuracy and confusion matrix for this fold
        accuracy = accuracy_score(y_test, y_pred_labels)
        conf_matrix = confusion_matrix(y_test, y_pred_labels)

        # Store results
        accuracy_scores.append(accuracy)
        confusion_matrices.append(conf_matrix)

        print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}")
        fold_no += 1

    # Aggregate results
    avg_accuracy = np.mean(accuracy_scores)
    avg_conf_matrix = np.mean(confusion_matrices, axis=0)

    # Print final results
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Confusion Matrix:\n{avg_conf_matrix}")

    return avg_accuracy, avg_conf_matrix


def main():
    """
         Main method to load and preprocess eeg data as per the BiHDM model
    :return:
    """
    # Set seed for reproducibility
    seed = 4
    np.random.seed(seed)
    tf.random.set_seed(seed)

    eeg_data, labels, subject_arr = load_processed_data()

    print(f'EEG Data Shape: {eeg_data.shape}')  # (Total trials, channels, time points)
    print(f'Labels Shape: {labels.shape}')  # (Total trials,)
    print(f'Subject Array Shape: {subject_arr.shape}')  # (Total Subjects)

    target_length = 64
    input_shape = (31, target_length)  # 31 electrodes, 64 time points
    # Generate subjects array for LOSO

    # Create the BiHDM model original
    print(f"BiHDM model 32 unit paper based")
    mean_acc, std_acc = evaluate_model_4_paper(eeg_data, labels, target_length, subject_arr, 50, input_shape)
    print(f"mean_acc: {mean_acc}, std_acc: {std_acc}")

    # Create the BiHDM model improved
    print(f"BiHDM model improved 64 unit analysis")
    # model = create_bi_hdm_model_improved(input_shape)
    # evaluate_model(eeg_data, labels, target_length, seed, 50, model)

    # Create the BiHDM model attn
    print(f"BiHDM model attn analysis")
    # model = create_bi_hdm_model_modified_attn(input_shape)
    # evaluate_model(eeg_data, labels, target_length, seed, 40, model)

    # Create the BiHDM model multi head attn
    print(f"BiHDM model multi head attn analysis")
    # model = create_bi_hdm_model_modified_multi_head_att(input_shape)
    # evaluate_model(eeg_data, labels, target_length, seed, 40, model)
    # k_fold_cross_validation(eeg_data, labels, 64, seed, 60)


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
