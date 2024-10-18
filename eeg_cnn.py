import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from eeg_mat_load import get_fixed_params, load_n_classify
from egg_cnn_models import create_cnn1d_model, create_cnn2d_model, create_cnn3d_model
from egg_cnn_models import reshape_eeg_data_for_cnn1d, reshape_eeg_data_for_cnn2d, reshape_eeg_data_for_cnn3d


# Frames: 24
# Depth/Channels: The number of EEG channels (62).
# Height/Max time window: The number of time windows (64).
# Width/Sampling: The number of features (5).


def perform_cnn(res_cnn1, res_cnn2, res_cnn3) -> None:
    """
    This method is to perform the CNN on the EEG data for 1D,2D and 3D classification.
    :param res_cnn1: The output file associate with CNN 1D classification.
    :param res_cnn2: The output file associate with CNN 2D classification.
    :param res_cnn3: The output file associate with CNN 3D classification.
    :return:
    """
    # Declare fixed input shapes for CNN 1-3
    input_shape_1d = (64, 62 * 5)
    input_shape_2d = (64, 62, 5)
    input_shape_3d = (64, 62, 5, 1)

    data_dirs, session_labels, eeg_keys = get_fixed_params()
    print(f" input_shape: {input_shape_1d}")
    res_cnn1.write(f"input_shape: {input_shape_1d}\n")
    res_cnn2.write(f"input_shape: {input_shape_2d}\n")
    res_cnn3.write(f"input_shape: {input_shape_3d}\n")

    model_cnn_1d = create_cnn1d_model(input_shape_1d)
    model_cnn_2d = create_cnn2d_model(input_shape_2d)
    model_cnn_3d = create_cnn3d_model(input_shape_3d)

    avg_loss_1d, avg_acc_1d = 0.0, 0.0
    avg_loss_2d, avg_acc_2d = 0.0, 0.0
    avg_loss_3d, avg_acc_3d = 0.0, 0.0

    count = 0
    for idx, data_dir in enumerate(data_dirs):
        # Get a list of all .mat files in the directory
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        # Iterate through each file and load the data
        for idx2, file in enumerate(mat_files):
            file_path = os.path.join(data_dir, file)
            eeg_data, session_label = load_n_classify(session_labels, eeg_keys, idx, file_path)

            # Print shapes to confirm
            # Should be (total_samples, 62, time_points)
            res_cnn1.write(f"EEG Data Shape: {eeg_data.shape}, Label Shape: {np.array(session_label).shape},")
            res_cnn2.write(f"EEG Data Shape: {eeg_data.shape}, Label Shape: {np.array(session_label).shape},")
            res_cnn3.write(f"EEG Data Shape: {eeg_data.shape}, Label Shape: {np.array(session_label).shape},")

            # W is no of time windows
            # Get the number of time windows (64 in this case)
            w_time = eeg_data.shape[2]

            # Model 1 performance
            eeg_data_reshaped = reshape_eeg_data_for_cnn1d(eeg_data, w_time)
            res_cnn1.write(f" EEG Data ReShape: {eeg_data_reshaped.shape},")
            test_loss, test_acc = evaluate_model(model_cnn_1d, np.array(session_label), eeg_data_reshaped, 8)
            res_cnn1.write(f" File: {(idx + 1)}'-'{file}, Accuracy: {test_acc}, Loss: {test_loss}\n")

            # Model 2 performance
            eeg_data_reshaped = reshape_eeg_data_for_cnn2d(eeg_data, w_time)
            res_cnn2.write(f" EEG Data ReShape: {eeg_data_reshaped.shape},")
            test_loss2, test_acc2 = evaluate_model(model_cnn_2d, np.array(session_label), eeg_data_reshaped, 32)
            res_cnn2.write(f" File: {(idx + 1)}'-'{file}, Accuracy: {test_acc2}, Loss: {test_loss2}\n")

            # Model 3 performance
            eeg_data_reshaped = reshape_eeg_data_for_cnn3d(eeg_data, w_time)
            res_cnn3.write(f" EEG Data ReShape: {eeg_data_reshaped.shape},")
            test_loss3, test_acc3 = evaluate_model(model_cnn_3d, np.array(session_label), eeg_data_reshaped, 32)
            res_cnn3.write(f" File: {(idx + 1)}'-'{file}, Accuracy: {test_acc3}, Loss: {test_loss3}\n")

            count = count + 1

            avg_loss_1d += test_loss
            avg_acc_1d += test_acc

            avg_loss_2d += test_loss2
            avg_acc_2d += test_acc2
            avg_loss_3d += test_loss3
            avg_acc_3d += test_acc3
            # evaluate_model_with_confusion_matrix(model_cnn_1d, np.array(session_label), eeg_data_reshaped, w_time)

    avg_loss_1d /= count
    avg_acc_1d /= count
    avg_loss_2d /= count
    avg_acc_2d /= count
    avg_loss_3d /= count
    avg_acc_3d /= count

    # model_cnn_1d.save('models/eeg/cnn1d_model.h5')
    # model_cnn_2d.save('models/eeg/cnn2d_model.h5')
    # model_cnn_3d.save('models/eeg/cnn3d_model.h5')

    res_cnn1.write(f"Average Test Accuracy CNN 1D : {avg_acc_1d}, Loss: {avg_loss_1d}\n")
    res_cnn2.write(f"Average Test Accuracy CNN 2D : {avg_acc_2d}, Loss: {avg_loss_2d}\n")
    res_cnn3.write(f"Average Test Accuracy CNN 3D : {avg_acc_3d}, Loss: {avg_loss_3d}\n")

    print(f"Average Test Accuracy: {avg_acc_1d}, Loss: {avg_loss_1d}\n")
    print(f"Average Test Accuracy: {avg_acc_2d}, Loss: {avg_loss_2d}\n")
    print(f"Average Test Accuracy: {avg_acc_3d}, Loss: {avg_loss_3d}\n")


def evaluate_model(model, labels, eeg_data, batch_size):
    """
    This method evaluates the performance of various CNN model
    :param model: The CNN model used for evaluation
    :param labels: The data labels initially provided for seed 4 based on experiment
    :param eeg_data: The eeg_data nd array reshaped as per model
    :param batch_size: The batch to be used for evaluation
    :return:
    """
    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=batch_size)

    # Train the model
    # history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    # print(f'Test accuracy: {test_acc}, loss: {test_loss}')
    return test_loss, test_acc


def evaluate_model_with_confusion_matrix(model, y_true, eeg_data_reshaped):
    """
    This method is to plot the confusion matrix with of various CNN models
    :param model: The CNN model used for evaluation
    :param y_true: The actual y value
    :param eeg_data_reshaped: The eeg_data reshaped as per model
    :return:
    """
    # Make predictions
    y_pred = model.predict(eeg_data_reshaped)
    y_pred = np.argmax(y_pred, axis=1)  # Get the predicted class labels

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Sad', 'Fear', 'Happy'],
                yticklabels=['Neutral', 'Sad', 'Fear', 'Happy'])
    plt.title("Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

    # Optionally print the classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Neutral', 'Sad', 'Fear', 'Happy']))


def main():
    # model = create_cnn3d_model(input_shape=(64, 62, 5, 1))
    # model.summary()
    with (open("results_cnn1d.txt", "w") as res_cnn1,
          open("results_cnn2d.txt", "w") as res_cnn2,
          open("results_cnn3d.txt", "w") as res_cnn3):
        perform_cnn(res_cnn1, res_cnn2, res_cnn3)


if __name__ == '__main__':
    main()
