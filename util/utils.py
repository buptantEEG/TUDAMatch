import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler
from shutil import copy
import matplotlib.pyplot as plt
# import plotly.express as px
import seaborn as sns
import umap
import umap.plot

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
            param.requires_grad = requires_grad

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def decomposition_UMAP_2D(all_data,save_path,labels,n_neighbors=5, min_dist=0.3):
    import umap
    
    plt.figure(figsize=(12,8))
    plt.title('Decomposition using UMAP 2D')
    colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    labels = labels
    
    for i,dataset in enumerate(all_data):
        data = dataset.x_data
        print(data.shape)
        umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2).fit_transform(data.reshape(data.shape[0],-1))
        plt.scatter(umap_data[:,0], umap_data[:,1], c = colors[i])
        
        # plt.scatter(umap_data[:,1], umap_data[:,2])
        # plt.scatter(umap_data[:,2], umap_data[:,0])
    plt.legend(labels=labels,title="classes")
    plt.savefig(f'{save_path}/umap_2D.png')

def decomposition_UMAP_3D(all_data,save_path,labels,n_neighbors=100, min_dist=0.3):
    import umap
    fig=plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')  #3d图需要加projection='3d'
    plt.title('Decomposition using UMAP 3D')
    colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    labels = labels
    
    for i,dataset in enumerate(all_data):
        data = dataset.x_data
        print(data.shape)
        umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3).fit_transform(data.reshape(data.shape[0],-1))
        plt.scatter(umap_data[:,0], umap_data[:,1],umap_data[:,2], c = colors[i], cmap='viridis')

        # plt.scatter(umap_data[:,1], umap_data[:,2])
        # plt.scatter(umap_data[:,2], umap_data[:,0])
    plt.legend(labels=labels,title="classes")
    plt.savefig(f'{save_path}/umap_3D.png')

def decomposition_UMAP_labels(all_data,source_encoder,target_encoder,save_path,labels,n_neighbors=100):
    save_path=f'{save_path}/umap_labels/'
    os.makedirs(save_path,exist_ok=True)
    for i,dataset in enumerate(all_data):
        _decomposition_UMAP_labels(dataset,source_encoder,save_path,n_neighbors=n_neighbors, type=labels[i],modeltype='sec')

    for i,dataset in enumerate(all_data):
        _decomposition_UMAP_labels(dataset,target_encoder,save_path,n_neighbors=n_neighbors, type=labels[i],modeltype='tgt')

def _decomposition_UMAP_labels(dataset,encoder,save_path,n_neighbors=100, type='origin',modeltype='None'):
    all_ys = dataset.y_data
    # all_ys = all_ys.tolist()
    # all_ys = all_ys.map({'W':0,'N1':1,'N2':2,'N3':3,'REM':4})
    print(all_ys.shape)
    labels = np.unique(all_ys)
    print(labels)
    # label_AASM = ['W','N1','N2','N3','REM']

    # plot
    plt.figure(figsize=(12,8)) #.add_subplot(111, projection='3d')  #3d图需要加projection='3d'
    plt.title('Decomposition using UMAP 3D with Labels')
    # colors = ['c', 'b', 'g', 'm', 'y', 'k', 'w']

    features = encode(encoder,dataset)
    umap_data = umap.UMAP(n_neighbors=n_neighbors, n_components=2).fit(features.reshape(features.shape[0],-1))
    # plt.scatter(umap_data[:,0], umap_data[:,1], s=20 , c = colors[i], alpha=0.3, cmap='viridis')
    umap.plot.points(umap_data,cmap='viridis',labels = all_ys)
        # plt.scatter(umap_data[:,1], umap_data[:,2])
        # plt.scatter(umap_data[:,2], umap_data[:,0])
    # plt.legend(labels=label_AASM,title="classes")
    plt.savefig(f'{save_path}/{type}_data_by_{modeltype}_model.png')

def encode(encoder,dataset):
    encoder.eval()
    outputs_list=[]
    with torch.no_grad():
            for data,_ in dataset:
                # data=data.
                data = data.to('cuda')
                pred_labels,(outputs,_) = encoder(data.unsqueeze(0))
                outputs_numpy = outputs.detach().cpu().numpy() # probs (AUROC)
                outputs_list.append(outputs_numpy)
            outputs_list = np.concatenate(outputs_list)
    return outputs_list

def copy_Files(destination,home_dir,encoder_name):
    destination_dir = os.path.join(destination, "backup_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy(f"{home_dir}/main.py", os.path.join(destination_dir, "main.py"))
    copy(f"{home_dir}/dataset/data_loaders.py", os.path.join(destination_dir, "data_loaders.py"))
    copy(f"{home_dir}/dataset/hyperparameters.py", os.path.join(destination_dir, f"hyperparameters.py"))
    copy(f"{home_dir}/dataset/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy(f"{home_dir}/model/encoder_{encoder_name}.py", os.path.join(destination_dir, f"encoder_{encoder_name}.py"))
    copy(f"{home_dir}/model/network.py", os.path.join(destination_dir, f"network.py"))
    # copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    # copy("models/TC.py", os.path.join(destination_dir, "TC.py"))


def get_nonexistant_path(fname_path):
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = f"{filename}_{i}"
    while os.path.exists(new_fname):
        new_fname = f"{filename}_{i + 1}"
        i+=1
    return new_fname