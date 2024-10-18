import os
from contextlib import redirect_stdout

import mne.io
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
import scipy.io as sio
import mne as eeg
import torchmetrics

import eeg_labels as eel
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

training_features = Tensor()
training_labels = Tensor()
value_features = Tensor()
value_labels = Tensor()

'''
Chrononet paper and model
  Title: ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification
  https://arxiv.org/abs/1802.00308
  Data collection from Emotiv EPOC+ 14-Channel Wireless EEG Headset
  Device
  Emotiv EPOC+ 14-Channel
  https://www.emotiv.com/products/epoc?srsltid=AfmBOoqg88Ig8eYKeN-b4G939UoCyz6jxEaSwQkaSfKCVtzpoSWMc_Ds
  
  Studying functional brain networks from dry electrode EEG set during music and resting 
  states in neuro development disorder
  https://www.biorxiv.org/content/10.1101/759738v1.full.pdf
'''


# cnn rnn

def gen_torch_random(channel=22, samples=15000):
    # batch size, channels
    out = torch.randn(3, channel, samples)
    # print(out.shape)
    return out


class Chrononet(nn.Module):
    def __init__(self, inplace):
        super(Chrononet, self).__init__()
        # conv_left 2 32 /2  conv_center 4 32 /2  conv_right 8 32 /2
        self.block1 = Block(14)
        self.block2 = Block(96)
        self.block3 = Block(96)
        self.relu = nn.ReLU()

        self.gru_layer1 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)
        self.gru_layer2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.gru_layer3 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.gru_layer4 = nn.GRU(input_size=96, hidden_size=32, batch_first=True)

        self.linear = nn.Linear(inplace['linear'], 1)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(32, 1)

    # CNN forward function
    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.block1(x))
        x = self.relu(self.block2(x))
        x = self.relu(self.block3(x))
        # print(x.shape)
        x = x.permute(0, 2, 1)

        print(f'Permuted x.shape {x.shape}')
        gru_out1, hh_out = self.gru_layer1(x)
        print(f'Shape of gru out 1: {gru_out1.shape}, hidden_out: {hh_out.shape}')

        # Feed layer one to layer 2 directly
        gru_out2, _ = self.gru_layer2(gru_out1)
        print(f'Shape of gru out 2: {gru_out2.shape}')

        # Feed joined layer one & layer 2 via tensor.cat to layer 3
        gru_cat_out_1_2 = torch.cat((gru_out1, gru_out2), 2)
        print(f'Shape of gru out cat of 1 & 2: {gru_cat_out_1_2.shape}')
        gru_out3, _ = self.gru_layer3(gru_cat_out_1_2)
        print(f'Shape of gru out 3: {gru_out3.shape}')

        # Feed joined layer 1 & layer 2 & layer 3 via tensor.cat to layer 4 but use a linear/dense function in between
        gru_cat_out_1_2_3 = torch.cat((gru_out1, gru_out2, gru_out3), 2)
        print(f'Shape of gru out cat of 1,2,3 : {gru_cat_out_1_2_3.shape}')

        # need to pass to a linear layer to reduce the last 1875 to 1
        linear_out = self.linear(gru_cat_out_1_2_3.permute(0, 2, 1))
        linear_out = nn.ReLU()(linear_out)
        print(f'Shape of linear out shape: {linear_out.shape}')

        # Need to reduce the 1875 from the concatenated from previous layer for gru 4
        gru_out4, _ = self.gru_layer4(linear_out.permute(0, 2, 1))
        print(f'Shape of gru out 4: {gru_out4.shape}')

        # Use activation function similar to softmax
        flatten_out = self.flatten(gru_out4)
        fcl_out = self.fcl(flatten_out)
        x = fcl_out
        return x


class Block(nn.Module):
    def __init__(self, inplace):
        super().__init__()
        # conv_left 2 32 /2  conv_center 4 32 /2  conv_right 8 32 /2
        self.conv1 = nn.Conv1d(inplace, 32, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(inplace, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(inplace, 32, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f'Block shapes {x}')
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        # print(x1.shape, x2.shape, x3.shape)
        x = torch.cat((x1, x2, x3), 1)

        return x


def read_id_data(path: str):
    subjects = []
    for i in glob(path + '*.mat'):
        data = sio.loadmat(i)
        # print(data.keys())
        data = data['clean_data']
        # print(data.shape)

        with redirect_stdout(open(os.devnull, 'w')):
            data = convert_mat_to_mne(data)
        subjects.append(data)
    return subjects


def convert_mat_to_mne(data, sampling_rate=128):
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                     'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    channel_types = ['eeg'] * 14
    info = eeg.create_info(channel_names,
                           ch_types=channel_types,
                           sfreq=sampling_rate)
    info.set_montage(montage='standard_1020')
    print('info', info)
    raw_data = mne.io.RawArray(data=data, info=info)
    raw_data.set_eeg_reference()
    raw_data.filter(l_freq=1, h_freq=30)

    # data shape is 10 columns and rows 1000 freq is 100Hz
    epochs = eeg.make_fixed_length_epochs(raw_data, duration=4, overlap=0)
    epoch_data = epochs.get_data()
    print(epoch_data.shape)
    return epoch_data


def accuracy(data_arr, label_arr, grp_arr):
    gkf = GroupKFold()
    accuracies = []
    init_out = gen_torch_random(14, 512)
    inplace = {'block1': 14, 'block2': 96, 'block3': 96, 'linear': 64}

    # original data (5744, 1250, 19)
    # reshape to (5744 x 1250, 19)
    # later revert to original (5744, 1250, 19)
    # print(f'Reshaped value train_features {train_features.shape}
    #   {train_features.reshape(-1, train_features.shape[-1]).shape}')
    for train_idx, val_idx in gkf.split(data_arr, label_arr, groups=grp_arr):
        train_features, train_labels = data_arr[train_idx], label_arr[train_idx]
        val_features, val_labels = data_arr[val_idx], label_arr[val_idx]

        scaler = StandardScalar3D()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        # move the axis as pytorch requires the channel in center and samples last
        train_features = np.moveaxis(train_features, 1, 2)
        val_features = np.moveaxis(val_features, 1, 2)
        # Convert to Tensor format
        train_features = torch.Tensor(train_features)
        train_labels = torch.Tensor(train_labels)
        val_features = torch.Tensor(val_features)
        val_labels = torch.Tensor(val_labels)
        print(f'shape for train features: {train_features.shape}, labels {train_labels.shape}')
        print(f'shape for value features: {len(val_features)}, labels {len(val_labels)}')
        break

    # print(f'train_features {train_features}')
    # print(f'val_features {val_features}')
    chronomodel = ChronoModel(inplace,
                              train_features=train_features,
                              train_labels=train_labels,
                              val_features=val_features,
                              val_labels=val_labels)
    chronomodel(init_out)
    global training_features, training_labels, value_features, value_labels
    training_features = train_features
    training_labels = train_labels
    value_features = val_features
    value_labels = val_labels
    #
    trainer = Trainer(max_epochs=1)
    trainer.fit(chronomodel)

    # result = np.mean(accuracies)
    # print('Accuracy: ', chronomodel.accuracy)


# Ref
# https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
class StandardScalar3D(BaseEstimator, TransformerMixin):
    def __init__(self, dim=3):
        self.scaler = StandardScaler()

    def fit(self, x, y=None):
        self.scaler.fit(x.reshape(-1, x.shape[2]))
        return self

    def transform(self, x):
        return self.scaler.transform(x.reshape(-1, x.shape[2])).reshape(x.shape)


class ChronoModel(LightningModule):
    def __init__(self, inplace,
                 train_features, train_labels,
                 val_features, val_labels):
        super(ChronoModel, self).__init__()
        self.model = Chrononet(inplace)
        self.lr = 1e-3
        self.bs = 12
        self.worker = 2
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels

        # print(f'train_features {train_features}')
        # print(f'val_features {val_features}')

    # def update(self, train_features, train_labels, val_features, val_labels):
    #     self.train_features = train_features
    #     self.train_labels = train_labels
    #     self.val_features = val_features
    #     self.val_labels = val_labels

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        dataset = TensorDataset(training_features, training_labels)
        print('train dataset', dataset)
        loader = DataLoader(dataset, batch_size=self.bs, num_workers=self.worker, shuffle=True)
        print('train loader', loader)
        return loader

    def training_step(self, batch, batch_idx):
        signal, label = batch
        out = self(signal.float())
        loss = self.criterion(out.flatten(), label.float().flatten())
        acc = self.accuracy(out.flatten(), label.long().flatten())
        return {'loss': loss, 'acc': acc}

    def trained_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean().detach().cpu().numpy().round(2)
        loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu().numpy().round(2)
        print('train acc loss', acc, loss)

    def val_dataloader(self):
        dataset = TensorDataset(value_features, value_labels)
        loader = DataLoader(dataset, batch_size=self.bs, num_workers=self.worker, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        signal, label = batch
        out = self(signal.float())
        loss = self.criterion(out.flatten(), label.float().flatten())
        acc = self.accuracy(out.flatten(), label.long().flatten())
        self.validation_step_outputs.append(acc)
        return {'loss': loss, 'acc': acc}

    def on_validation_epoch_end(self):
        print(f'output: {self.validation_step_outputs}')
        epoch_acc_average = torch.stack(self.validation_step_outputs).mean()
        print(f'Validation Accuracy: {epoch_acc_average}')
        self.validation_step_outputs.clear()


def main():
    init_out = gen_torch_random(14, 512)
    print(init_out.shape)

    # inplace = {'block1': 22, 'block2': 96, 'block3': 96, 'linear': 1875}
    inplace = {'block1': 14, 'block2': 96, 'block3': 96, 'linear': 64}
    # chrononet = Chrononet(inplace)
    # result = chrononet(init_out)
    # print(f'result shape {result.shape}')
    # print(result)

    idd_subjects = read_id_data("data/chrononet/clean_data/IDD/Rest/")
    tdc_subjects = read_id_data("data/chrononet/clean_data/TDC/Rest/")
    print(f'idd_subjects: {len(idd_subjects)}')
    print(f'tdc_subjects: {len(tdc_subjects)}')
    #
    data_arr = eel.create_labels(tdc_subjects, idd_subjects)
    print(f'data_arr: {len(data_arr)}')
    #
    data_arr[0] = np.moveaxis(data_arr[0], 1, 2)
    accuracy(data_arr[0], data_arr[1], data_arr[2])


if __name__ == '__main__':
    main()
