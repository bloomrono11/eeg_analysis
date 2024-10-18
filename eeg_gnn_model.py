import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import compute_class_weight
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, SAGEConv

from eeg_mat_load import load_n_classify, get_fixed_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eeg_channels = 62
max_time_window = 64


# Define the GNN Sage Conv model
class EEGEmotionGNNSAGE(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(EEGEmotionGNNSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.dropout(x)

        # Global mean pooling, use batch information for pooling
        x = global_mean_pool(x, batch)

        # Final fully connected layer/ Classification layer
        x = self.fc(x)
        return x


def create_graph(eeg_data, label, edge_index):
    """
      Function to create graph data from EEG data
    :param eeg_data: The data loaded from the mat files
    :param label: The session labels associated with the data
    :param edge_index: The edge index calculated via the edge function
    :return:
    """
    # eeg_data shape: (D, W, S) = (62, 64/60, 5)
    # Reshape eeg_data to (W, D*S) for features
    # print(f"Original data shape: {eeg_data.shape}")

    # Get the original number of time windows
    original_time_windows = eeg_data.shape[1]  # This will be W

    # Check if padding is needed
    if original_time_windows < max_time_window:
        # Pad the node features to max_time_windows (64) => Shape: (62, 4, 5) => Shape: (62, 64, 5)
        padding = torch.zeros((eeg_channels, max_time_window - original_time_windows, 5), dtype=torch.float32)
        padded_node_features = torch.cat((torch.tensor(eeg_data, dtype=torch.float32), padding), dim=1)
    else:
        padded_node_features = torch.tensor(eeg_data, dtype=torch.float32)  # no padding needed

    x = padded_node_features.reshape(max_time_window, eeg_channels * 5)  # Shape: (64/60, 310)
    y = torch.tensor([label], dtype=torch.long)  # Graph label
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    return graph_data


def create_edge_index_k_near():
    """
    Create an edge index based on the k-nearest neighbors of the EEG channels.

    :return: A tensor of shape [2, num_edges] representing the edges
    """
    # Assuming you have some way to define positions for the channels
    # Here we'll use a random placeholder; replace with actual channel positions.
    positions = np.random.rand(eeg_channels, 2)  # Replace this with your actual positions
    knn_graph = kneighbors_graph(positions, n_neighbors=4, mode='connectivity', include_self=False)

    edges = []
    for i in range(knn_graph.shape[0]):
        for j in range(knn_graph.shape[1]):
            if knn_graph[i, j] > 0:  # There is an edge between node i and node j
                edges.append((i, j))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def perform_gnn() -> None:
    """
    This method is to perform GNN for the entire dataset loaded one file at a time
    loads 3 different experimental datasets of 15 subjects and 4 classes respectively
    Label: The labels of the three sessions for the same subjects are as follows,
       session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
       session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
       session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
       The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.
    :return:
    """

    # Fetches the predefined values for data directory, session label and eeg keys
    data_dirs, session_labels, eeg_keys = get_fixed_params()

    # Initialize model, loss, and optimizer and 4 classes: 0, 1, 2, 3
    model = EEGEmotionGNNSAGE(num_node_features=eeg_channels * 5, hidden_channels=128, num_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Adjust learning rate every 50 epochs
    train_avg_acc, train_avg_loss = 0, 0
    test_avg_acc, test_avg_loss = 0, 0

    # For storing cumulative confusion matrix
    cumulative_conf_matrix = np.zeros((4, 4), dtype=int)
    for idx, data_dir in enumerate(data_dirs):
        mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        print(f"Processing Mat file index {idx + 1}")
        mat_file_len = len(mat_files)
        for idx2, filename in enumerate(mat_files):
            file_path = os.path.join(data_dir, filename)
            eeg_data, session_label = load_n_classify(session_labels, eeg_keys, idx, file_path)

            # Create the dataset
            graphs = prepare_dataset(eeg_data, session_label)
            # Create DataLoaders for both train and test datasets
            train_loader, test_loader = prepare_data_loaders(graphs, test_size=0.2, batch_size=20)
            # Training Loop
            for epoch in range(1, 100):
                # Start training
                train_loss, train_acc = train(model, train_loader, optimizer, criterion)
                scheduler.step()  # Adjust learning rate
                train_avg_acc += train_acc
                train_avg_loss += train_loss
            # Start testing
            test_acc, test_loss, conf_matrix = test_eval(model, test_loader, criterion)
            test_avg_acc += test_acc
            test_avg_loss += test_loss
            cumulative_conf_matrix += conf_matrix
        # After training and testing all files for the current data directory (session)
        train_avg_acc = train_avg_acc / mat_file_len
        train_avg_loss = train_avg_loss / mat_file_len
        print(f" Train avg Acc & loss after session {idx + 1}: {train_avg_acc:.4f},{train_avg_loss:.4f}")
        print(f" Test avg Acc & loss after session {idx + 1}: {test_avg_acc:.4f},{test_avg_loss:.4f}")
        print(f"Cumulative Confusion Matrix after session {idx + 1}:\n{cumulative_conf_matrix}")
        train_avg_loss = 0
        train_avg_acc = 0
        test_avg_acc = 0
        test_avg_loss = 0

    plot_confusion_matrix(cumulative_conf_matrix)


def prepare_dataset(eeg_sessions, labels):
    """
    This method is to convert the loaded data into the format expected by GNN model
    :param eeg_sessions: eeg_data passed into GNN model
    :param labels:  the session labels initialized at start
    :return:
    """
    edge_index = create_edge_index_k_near()  # Use the previous edge index function
    graphs = []

    for eeg_data, label in zip(eeg_sessions, labels):
        # Ensure the data shape is correct (62, 64, 5)
        graph = create_graph(eeg_data, label, edge_index)
        # print(f"Original Shape x: {eeg_data.shape}")
        # print(f"Shape x: {x.shape}, edge_index: {edge_index.shape}")
        graphs.append(graph)

    return graphs


def prepare_data_loaders(graphs, test_size, batch_size):
    """
    This function splits the graphs into train and test datasets and returns DataLoaders for both.

    :param graphs: List of graph data
    :param test_size: Proportion of the dataset to use as the test set (default is 20%)
    :param batch_size: Batch size for DataLoader
    :return: train_loader, test_loader
    """
    # Determine the sizes for train and test sets
    total_size = len(graphs)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(graphs, [train_size, test_size])

    # Create DataLoaders for both train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print(f"Graph x {graphs[0].x.shape},
    # edge x {graphs[0].edge_index.shape}, train_loader: {train_loader.batch_size}")
    return train_loader, test_loader


# Training loop
def train(model, train_loader: DataLoader, optimizer, criterion):
    model.to(device)
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Calculate loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)  # Get predicted class
        correct += (pred == data.y).sum().item()  # Track correct predictions

    accuracy = correct / len(train_loader.dataset)  # Calculate accuracy
    loss = total_loss / len(train_loader.dataset)  # Calculate loss

    return loss, accuracy


# Testing loop
def test_eval(model, test_loader: DataLoader, criterion) -> tuple[float, float, any]:
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            data = data.to(device)
            labels = data.y.to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, labels)
            total_loss += loss.item()

            predicted = output.argmax(dim=1)  # Get predicted class

            # Compare predictions with true labels
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())  # Store predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())  # Store true labels for confusion matrix

    avg_loss = total_loss / total
    avg_acc = correct / total
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    return avg_acc, avg_loss, conf_matrix


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Neutral', 'Sad', 'Fear', 'Happy'],
                yticklabels=['Neutral', 'Sad', 'Fear', 'Happy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    perform_gnn()


if __name__ == '__main__':
    main()
