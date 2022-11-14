from datetime import datetime
import time
import os
# from model import discriminator
import model
import re
import argparse
import numpy as np
import random

from dataset.data import get_dataloaders
from dataset.hyperparameters import adamatch_hyperparams
from dataset.data_loaders import data_generator_augment_da, data_generator_noaugment, data_generator_tudamatch_random

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from util import writer_func
from util.utils import _logger, copy_Files, get_nonexistant_path,setup_seed
from util.train_test import *


class Config():
    def __init__(self,
                 # channels
                 ):
        self.input_channels = 1
        self.final_out_channels = 128

        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 127
        self.afr_reduced_cnn_size = 2
        self.d_model = 48
        self.inplanes = 2
        self.nhead = 4
        self.num_layers = 1

        self.batch_size = 256
        self.num_classes = 5

# get source and target data
# data = get_dataloaders("./", batch_size_source=32, workers=2)


# source_path = f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/'
# target_path = f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/'
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')

parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')

parser.add_argument('--encoder', default='MMASleepNet_EEG', type=str,
                    help='Encoder model name')

parser.add_argument('--seed', default=0, type=int,
                    help='seed value')

parser.add_argument('--logs_save_dir', default='../experiments_save/', type=str,
                    help='saving directory')

parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')

parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--target_path', default=f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/', type=str,
                    help='Target data path')

parser.add_argument('--source_path', default=f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/', type=str,
                    help='Source data path')

parser.add_argument('--train_mode', default=f'TUDAMatch', type=str,
                    help='TUDAMatch')

parser.add_argument('--discriminator', default=f'Discriminator_ATT', type=str,
                    help='Discriminator_ATT or Discriminator_AR or Discriminator')

args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
run_description = args.run_description
encoder = args.encoder
discriminator = args.discriminator
save_dir = os.path.join('../',args.logs_save_dir)
os.makedirs(save_dir, exist_ok=True)

train_mode = args.train_mode
SEED = args.seed
setup_seed(SEED)

# 创建log和备份文件夹
experiment_log_dir = os.path.join(save_dir, f'{train_mode}_{encoder}'+experiment_description, run_description+f"_seed_{SEED}")
# experiment_log_dir=get_nonexistant_path(experiment_log_dir)

model_save_dir = os.path.join(experiment_log_dir, 'model_save')
history_save_dir = os.path.join(experiment_log_dir, 'histories')
tensorboard_log_dir = os.path.join(experiment_log_dir, 'tensorboard')
log_dir = os.path.join(experiment_log_dir, 'logs')

os.makedirs(experiment_log_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(history_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

# 备份文件
copy_Files(experiment_log_dir, home_dir, encoder_name=encoder)

target_path = args.target_path
source_path = args.source_path

now= datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

log_file_name = os.path.join(
    log_dir, f"log_{now}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Source Path: {source_path}')
logger.debug(f'Target Path:  {target_path}')
logger.debug(f'Model:    {encoder}')
logger.debug("=" * 45)

source_files = os.listdir(source_path)
source_files.sort(key=lambda x: int(str(re.findall("\d+", x)[0])))
target_files = os.listdir(target_path)
target_files.sort(key=lambda x: int(str(re.findall("\d+", x)[0])))
source_path_ = []
target_path_ = []
for file in source_files:
    source_path_.append(source_path+file)
for file in target_files:
    target_path_.append(target_path+file)
# print(path)
logger.debug(f"source_files_len: {len(source_path_)}")
logger.debug(f"target_files_len: {len(target_path_)}")

source_files = source_path_[:10]
target_files = target_path_[:10]

logger.debug(f"source_files_len_use: {len(source_files)}")
logger.debug(f"target_files_len_use: {len(target_files)}")

configs = Config()

if encoder == 'MMASleepNet_EEG':
    configs.features_len = 128
    configs.batch_size = 32

if discriminator != 'None':
    configs.feat_dim = 3072
    configs.att_hid_dim = 512
    configs.patch_size = 128
    configs.out_channels = configs.final_out_channels
    configs.depth = 8
    configs.heads = 4
    configs.mlp_dim = 64
    # GRU configs
    configs.disc_n_layers = 1
    configs.disc_AR_hid = 512
    configs.disc_AR_bid = False
    configs.disc_hid_dim = 100
    configs.disc_out_dim = 1

# Tensorboard
writer = SummaryWriter(log_dir=tensorboard_log_dir)
writer_func.save_config(writer, configs)

n_classes = 5
start_time = time.time()


if train_mode == 'TUDAMatch':
    source_epochs = 100
    target_epochs = 1000
    # target_epochs = 500

    data = data_generator_tudamatch_random(source_files, target_files, batch_size=configs.batch_size, workers=0, logger=logger)
    source_loaders_strong = data[0]
    source_loaders_weak = data[1]
    target_loaders = data[2]

    logger.debug("Data loaded ...")

    hparams = adamatch_hyperparams()

    source_model = eval(f'model.{encoder}(config=configs)')
    target_model = eval(f'model.{encoder}(config=configs)')
    feature_discriminator = eval(f'model.{discriminator}(configs=configs)')

    tudamatch = model.TUDAMatch(source_model=source_model, 
                        target_model= target_model, 
                        feature_discriminator= feature_discriminator,
                        source_epochs = source_epochs, 
                        target_epochs = target_epochs, 
                        logger=logger, 
                        writer=writer)

    target_model = tudamatch.run(source_loaders_strong,source_loaders_weak, target_loaders, hparams, experiment_log_dir)
    
    # evaluate the model
    pic_path=tudamatch.plot_metrics(experiment_log_dir, now)

    # returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
    tudamatch.plot_cm_roc(target_loaders[1], pic_path, now)


    
end_time = time.time()
logger.debug(f'time {(end_time-start_time)/60} min')

writer.close()
logger.debug("Done!")

