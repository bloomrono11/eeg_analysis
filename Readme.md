# What is this repo about?

Understand and evaluating various eeg algorithms from papers for seed IV dataset

## Eeg analysis
 Other python files exist which showcase how to perform ML models and learning using various eeg data and stuff
 **todo**: organize the file structure

## Seed 4 dataset

Details regarding Seed 4 data [Link](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html)
Subjects: 15
Channels: 62
Trails for each subject: 24
Emotion Labels/Classifications/number of classes: 4
male:1、2、6、7、12、13
female:3、4、5、8、9、10、11、14、15

Label:
The labels of the three sessions for the same subjects are as follows,
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];

The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.

Trails for each subject 24
Channels 64

### Seed 4 raw
Data keys from mat files
the xx is replaced with each subject 2 charater initials
EEG Data keys: dict_keys(['__header__', '__version__', '__globals__',
 'xx_eeg1', 
 'xx_eeg2', 
 'xx_eeg3', 
 'xx_eeg4', 
 'xx_eeg5', 
 'xx_eeg6', 
 'xx_eeg7', 
 'xx_eeg8', 
 'xx_eeg9', 
 'xx_eeg10', 
 'xx_eeg11', 
 'xx_eeg12', 
 'xx_eeg13', 
 'xx_eeg14', 
 'xx_eeg15', 
 'xx_eeg16', 
 'xx_eeg17',
 'xx_eeg18', 
 'xx_eeg19', 
 'xx_eeg20', 
 'xx_eeg21',
 'xx_eeg22', 
 'xx_eeg23', 
 'xx_eeg24'

### Seed 4 Smooth
Data keys from mat files
EEG Data keys: dict_keys(['__header__', '__version__', '__globals__',
 'de_movingAve1', 'de_LDS1', 'psd_movingAve1', 'psd_LDS1',
 'de_movingAve2', 'de_LDS2', 'psd_movingAve2', 'psd_LDS2',
 'de_movingAve3', 'de_LDS3', 'psd_movingAve3', 'psd_LDS3', 
 'de_movingAve4', 'de_LDS4', 'psd_movingAve4', 'psd_LDS4', 
 'de_movingAve5', 'de_LDS5', 'psd_movingAve5', 'psd_LDS5',
 'de_movingAve6', 'de_LDS6', 'psd_movingAve6', 'psd_LDS6',
 'de_movingAve7', 'de_LDS7', 'psd_movingAve7', 'psd_LDS7',
 'de_movingAve8', 'de_LDS8', 'psd_movingAve8', 'psd_LDS8',
 'de_movingAve9', 'de_LDS9', 'psd_movingAve9', 'psd_LDS9',
 'de_movingAve10', 'de_LDS10', 'psd_movingAve10', 'psd_LDS10',
 'de_movingAve11', 'de_LDS11', 'psd_movingAve11', 'psd_LDS11',
 'de_movingAve12', 'de_LDS12', 'psd_movingAve12', 'psd_LDS12',
 'de_movingAve13', 'de_LDS13', 'psd_movingAve13', 'psd_LDS13',
 'de_movingAve14', 'de_LDS14', 'psd_movingAve14', 'psd_LDS14',
 'de_movingAve15', 'de_LDS15', 'psd_movingAve15', 'psd_LDS15',
 'de_movingAve16', 'de_LDS16', 'psd_movingAve16', 'psd_LDS16',
 'de_movingAve17', 'de_LDS17', 'psd_movingAve17', 'psd_LDS17',
 'de_movingAve18', 'de_LDS18', 'psd_movingAve18', 'psd_LDS18',
 'de_movingAve19', 'de_LDS19', 'psd_movingAve19', 'psd_LDS19',
 'de_movingAve20', 'de_LDS20', 'psd_movingAve20', 'psd_LDS20',
 'de_movingAve21', 'de_LDS21', 'psd_movingAve21', 'psd_LDS21',
 'de_movingAve22', 'de_LDS22', 'psd_movingAve22', 'psd_LDS22',
 'de_movingAve23', 'de_LDS23', 'psd_movingAve23', 'psd_LDS23',
 'de_movingAve24', 'de_LDS24', 'psd_movingAve24', 'psd_LDS24'