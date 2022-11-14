from sklearn import preprocessing
from scipy.stats import zscore
import numpy as np

# compute in the first column

def min_max_scaler(data):
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def standard_scaler(data): # 1d or 2d
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def max_abs_scaler(data):
    scaler = preprocessing.MaxAbsScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def normalizer(data):
    scaler = preprocessing.Normalizer()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

if __name__=='__main__':
    import sys
    sys.path.append("/disk1/brain/Sleep/code/")
    from dataset.SleepDataset import SleepDataset
    from torch.utils.data import DataLoader
    dataset = SleepDataset(task='children', downsample=100, data_path='../original_data/children/', process=min_max_scaler)
    dataloader = DataLoader(dataset, batch_size=60)
    train_data_sample, train_label_sample = next(iter(dataloader))
    a = train_data_sample[0]
    print(a[0].mean())
    print(train_data_sample[0])
    print('Done!')