import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from util.utils import setup_seed

class LoadDataset_from_numpy_augment(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,augment_type=None):
        super(LoadDataset_from_numpy_augment, self).__init__()
        self.augment_type = augment_type
        
        X_train = np.load(np_dataset[0])["x"]

        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:

            temp_data=np.load(np_file)["x"]
            if temp_data.shape[2] == X_train.shape[2]:
                X_train = np.vstack((X_train, temp_data))
            else: 
                continue

            y_train = np.hstack((y_train, np.load(np_file)["y"]))

        # isruc : ["F3_A2", "C3_A2", "F4_A1", "C4_A1", "O1_A2", "O2_A1", "ROC_A1", "X1"]
        # sleepedf : [fpz-cz, pz-oz, eog, emg]
        # shhs : [c3, c4, eog, emg]
        if "isruc" in np_dataset[0]:
            X_train = X_train[:,1,:][:,np.newaxis,:]

        elif "sleepedf" in np_dataset[0]:
            X_train = X_train.transpose(0,2,1)[:,0,:][:,np.newaxis,:]
            X_train=X_train 

        elif "shhs" in np_dataset[0]:
            X_train = X_train.transpose(0,2,1)[:,0,:][:,np.newaxis,:]
    

        
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        self.len = self.x_data.shape[0]

        if self.augment_type =='weak' or self.augment_type =='strong':
            self.x_data = DataTransform(self.x_data,self.augment_type)

    def __getitem__(self, index):

        return  self.x_data[index],self.y_data[index]


    def __len__(self):
        return self.len



def data_generator_tudamatch_random(source_files,target_files,batch_size,workers=0,logger='None',random_seed=0):

    setup_seed(random_seed)
    source_dataset_strong = LoadDataset_from_numpy_augment(source_files,augment_type='strong')
    source_dataset_weak = LoadDataset_from_numpy_augment(source_files,augment_type='weak')

    setup_seed(random_seed)
    target_dataset_strong = LoadDataset_from_numpy_augment(target_files,augment_type='strong')
    target_dataset_weak = LoadDataset_from_numpy_augment(target_files,augment_type='weak')


    train_size = int(len(source_dataset_strong) * 0.6)
    valid_size = int(len(source_dataset_strong) * 0.2)
    test_size = len(source_dataset_strong) - train_size - valid_size

    setup_seed(random_seed)
    source_train_dataset_strong,source_valid_dataset_strong,source_test_dataset_strong =  torch.utils.data.random_split(source_dataset_strong, [train_size, valid_size, test_size])
    setup_seed(random_seed)
    source_train_dataset_weak,source_valid_dataset_weak,source_test_dataset_weak =  torch.utils.data.random_split(source_dataset_weak, [train_size, valid_size, test_size])


    source_train_loader_strong = DataLoader(dataset=source_train_dataset_strong, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
    source_valid_loader_strong = DataLoader(dataset=source_valid_dataset_strong, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
    source_test_loader_strong = DataLoader(dataset=source_test_dataset_strong, batch_size=batch_size,shuffle=False, drop_last=False,num_workers=workers) 

    source_train_loader_weak = DataLoader(dataset=source_train_dataset_weak, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
    source_valid_loader_weak = DataLoader(dataset=source_valid_dataset_weak, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
    source_test_loader_weak = DataLoader(dataset=source_test_dataset_weak, batch_size=batch_size,shuffle=False, drop_last=False,num_workers=workers)                                                                                  
                                                                                 
    
    target_strong_loader = DataLoader(dataset=target_dataset_strong, batch_size=batch_size,shuffle=False, drop_last=False,num_workers=workers)   
    target_weak_loader = DataLoader(dataset=target_dataset_weak, batch_size=batch_size,shuffle=False, drop_last=False,num_workers=workers)   

    return (source_train_loader_strong, source_valid_loader_strong, source_test_loader_strong),(source_train_loader_weak, source_valid_loader_weak, source_test_loader_weak),(target_strong_loader,target_weak_loader)


