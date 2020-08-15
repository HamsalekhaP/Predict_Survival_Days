# Common constants and hyper-parameter settings
import torch


# path config
DATASET_PATH = 'regDataset'  # This is original dataset path.Use 'dummyDataset' to run on small dataset
# z-score normalised dataset path
NORM_DATASET_PATH = 'normDataset'  # Use 'dummyDataset' to run on small dataset
SAVE_WEIGHTS_PATH = 'checkpoints'  # path to save model and weights


# system config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE_COUNT = torch.cuda.device_count()


# DNN model config
Lr = 0.01
cli_args = {
}


# Model toggle features
MRI_TYPES = ['t1', 't1ce', 't2', 'flair']
SEG_LABELS = [1, 2, 4]  # annotations available on the segmentation mask
# use this specific modality as input
USE_ONE_MRI_TYPE = 't1'
# toggle this to use each of the MRI sequences associated with every patient as a data point
USE_ALL_MRI_TYPES = False
# toggle this to add append age to fully connected layer input
FEED_AGE = True
# toggle this to stack all MRI modalities as channels
USE_AS_CHANNELS = True
