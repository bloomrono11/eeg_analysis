# How to set up environment
  This project was set up to run with pycharm using a virtual environment
  Use the settings to go to python interpreter and setup a python.exe location
  Then create a new virtual environment

## Requirements
  Python 3.10.10
  * tensorflow 2.11.0
  * keras 2.11.0
  * pandas 2.1.0
  * numpy 1.26.0
  * scikit-learn 1.4.2
  * seaborn 0.13.2
  * mne 1.7.0
  * torch 2.1.0
  * torch-lightning 2.0.0
  * torch_geometric 2.1.0
  * torchmetrics 1.4.3

# How to extract the files
  You need to download the zip file provided by [Seed 4 Link](https://bcmi.sjtu.edu.cn/~seed/downloads.html#seed-iv-access-anchor)
  Need to request access to the dataset via this link
  [Download Link](https://bcmi.sjtu.edu.cn/~seed/downloads.html#seed-iv-access-anchor)
  Extract the eeg raw folder under data -> eeg -> eeg_raw_data same
  you final folder structure would be
  data
    eeg
      eeg_raw_data -> 1 -> 1_20160518.mat
      eeg_raw_data -> 1 -> 2_20150915.mat
      eeg_raw_data -> 2 -> 4_20151118.mat
      eeg_raw_data -> 2 -> 5_20160413.mat
      eeg_raw_data -> 3 -> 11_20151011.mat
      eeg_raw_data -> 3 -> 12_20150807.mat