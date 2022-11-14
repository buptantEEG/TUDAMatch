import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from util import writer_func
import os
from util.utils import set_requires_grad, get_nonexistant_path
from collections import Counter
from torch.utils.data import *
import sys
import copy

from collections.abc import Iterable
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class TUDAMatch():

    def __init__(self, source_model, target_model, feature_discriminator,source_epochs,target_epochs, logger,writer,):
        """
        NOTE: the actual AdaMatch paper doesn't separate between encoder and classifier,
        but I find it more practical for the purposes of setting up the networks.

        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger=logger
        self.writer=writer

        self.source_model = source_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.feature_discriminator = feature_discriminator.float().to(self.device)
        self.epochs_source = source_epochs
        self.epochs_target = target_epochs
        # self.feature_discriminator = Discriminator_ATT(configs).float().to(self.device)




    def run(self, source_loaders_strong, source_loaders_weak,target_loaders, hyperparams, save_path):
        """
        Trains the model (encoder + classifier).

        Arguments:
        ----------
        source_dataloader: PyTorch DataLoader
            DataLoader with source domain training, valid and test data .

        target_dataloader: PyTorch DataLoader
            DataLoader with target domain data.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.

        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.

        save_path: str
            Path to store model weights.

        Returns:
        --------
        target model: PyTorch neural network
            Neural network that receives an array of signal X, extract features F and classifies them into 5 classes.

        discriminator: PyTorch neural network
            Neural network that receives an array of features F and classifies it into 2 domain.
        """
        # Source data
        source_train_dataloader_srtong, source_valid_dataloader_strong, source_test_dataloader_strong = source_loaders_strong
        source_train_dataloader_weak, source_valid_dataloader_weak, source_test_dataloader_weak = source_loaders_weak
        # source_train_dataloader, source_valid_dataloader, source_test_dataloader = zip(source_train_dataloader_srtong,source_train_dataloader_weak
        #     ),zip(source_valid_dataloader_strong,source_valid_dataloader_weak),zip(source_test_dataloader_strong,source_test_dataloader_weak)
        # Target data
        target_dataloader_strong, target_dataloader_weak = target_loaders

        ckp_path = os.path.join(save_path,'model_save','ckpt_source_model.pt')
        his_path_target = os.path.join(save_path,'histories','history_target.npy')
        his_path_source = os.path.join(save_path,'histories','history_source.npy')

        # check if source only model exists, else train it ...
        if os.path.exists(ckp_path):
            if os.path.exists(his_path_source):
                self.history_source = np.load(his_path_source,allow_pickle=True).item()
            else:
                self.history_source = {'source_loss': [], 'accuracy_source_train': [], 'accuracy_source_valid': [], 'accuracy_source_test':[]}
            src_chkpoint = torch.load(ckp_path)['source_model_weights']
        else:

            self.Trainer(source_train_dataloader_srtong,source_train_dataloader_weak,source_valid_dataloader_strong,source_valid_dataloader_weak,source_test_dataloader_strong,source_test_dataloader_weak, hyperparams, ckp_path, his_path_source)
            src_chkpoint = torch.load(ckp_path)['source_model_weights']

        # Load trained mode;
        self.source_model.load_state_dict(src_chkpoint)
        self.target_model.load_state_dict(src_chkpoint)

        # Freeze the source domain model
        set_requires_grad(self.source_model, requires_grad=False)

        # configure hyperparameters
        lr = hyperparams['learning_rate_target']
        lr_dis = hyperparams['learning_rate_discriminator']
        wd = hyperparams['weight_decay']
        step_scheduler = hyperparams['step_scheduler']
        tau = hyperparams['tau']
        self.logger.debug(f"lr_target:{lr};lr_dis:{lr_dis}; wd:{wd};")

        iters = len(target_loaders[0])
        current_step = 0
        
        # #Freeze
        freeze = ['linear1','linear2','AFR','transformer_encoder']#,'transformer_encoder']#'EEG_feature','linear1','linear2','AFR','transformer_encoder'

        freeze_by_names(self.target_model, freeze)


        # # configure optimizer and scheduler
        optimizer_encoder = optim.Adam(self.target_model.parameters(),betas=(0.5,0.99), lr=lr, weight_decay=wd)
        optimizer_disc = optim.Adam(self.feature_discriminator.parameters(),betas=(0.5,0.99), lr=lr_dis, weight_decay=wd)
        
        if step_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=20)

        criterion_disc = nn.BCELoss()

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 30
        bad_epochs = 0
        source_t = 20
        adv_t = 10
        ada_t = 1

        self.logger.debug(f"source_t:{source_t};adv_t:{adv_t};ada_t:{ada_t};")

        self.history_target = {'acc_disc':[], 'target_loss': [],  'accuracy_target': []}


        epoch_accuracy_source, _, _, _ = self.evaluate_two(self.source_model,zip(source_test_dataloader_strong,source_test_dataloader_weak),return_lists_roc=True)
        target_acc_srcmodel, labels_list_target, _, pred_list_target = self.evaluate(self.target_model,target_dataloader_weak,return_lists_roc=True)


        # training loop
        for epoch in range(start_epoch, self.epochs_target):
            running_loss = 0.0


            # set network to training mode
            self.source_model.eval()
            self.target_model.train()
            # self.target_model.apply(fix_bn) 
            self.feature_discriminator.train()

            disc_labels=[]
            disc_pred=[]
            

            # dataset = zip(source_train_dataloader,target_dataloader_strong,target_dataloader_weak)
            dataset = zip(source_train_dataloader_srtong,source_train_dataloader_weak,target_dataloader_strong,target_dataloader_weak)
            # target_dataset=  zip(target_dataloader_strong,target_dataloader_weak)
            
            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for (source_data_strong, source_strong_labels),(source_data_weak, source_weak_labels) ,(target_strong_data, target_strong_labels), (target_weak_data, target_weak_labels) in dataset:
                source_data_strong, source_strong_labels,source_data_weak,source_weak_labels = source_data_strong.type(torch.FloatTensor).to(self.device), source_strong_labels.type(
                    torch.LongTensor).to(self.device), source_data_weak.type(torch.FloatTensor).to(self.device), source_weak_labels.type(torch.LongTensor).to(self.device),
                target_strong_data, target_strong_labels, target_weak_data, target_weak_labels = target_strong_data.float().to(self.device),target_strong_labels.float().to(
                        self.device),target_weak_data.float().to(self.device),target_weak_labels.float().to(self.device)

                
                #######################
                # train discriminator #
                #######################

                # zero gradients
                optimizer_disc.zero_grad()

                # pass data  through the model network.
                source_pred_strong, (source_latent_strong,source_feat_strong) = self.source_model(source_data_strong)
                source_pred_weak, (source_latent_weak,source_feat_weak) = self.source_model(source_data_weak)

                # pass data through the target model network.
                # pred_source_tgtmodel, (source_latent_tgtmodel,source_feat_tgtmodel) = self.target_model(source_data)
                pred_source_tgtmodel_strong, (source_latent_tgtmodel_strong,source_feat_tgtmodel_strong) = self.target_model(source_data_strong)
                pred_source_tgtmodel_weak, (source_latent_tgtmodel_weak,source_feat_tgtmodel_weak) = self.target_model(source_data_weak)
                
                pred_target_strong, (target_latent_strong,target_feat_strong) = self.target_model(target_strong_data)
                pred_target_weak, (target_latent_weak,target_feat_weak) = self.target_model(target_weak_data)

                # concatenate source and target features
                feat_concat = torch.cat((source_feat_strong, source_feat_weak,target_feat_strong,target_feat_weak), dim=0)
                # feat_concat = torch.cat((source_latent, target_latent_strong, target_latent_weak), dim=0)

                # predict the domain label by the discirminator network
                pred_concat = self.feature_discriminator(feat_concat.detach())
                # print(pred_concat.squeeze())
                # pred = np.argmax(pred_concat, axis=1)

                # prepare real labels for the training the discriminator
                label_src = torch.ones(source_feat_strong.size(0)*2).to(self.device)
                label_tgt = torch.zeros(target_feat_weak.size(0)*2).to(self.device)

                label_concat = torch.cat((label_src, label_tgt), 0)
                loss_disc = criterion_disc(pred_concat.squeeze(), label_concat.float())

                loss_disc.backward()

                # Update disciriminator optimizer
                optimizer_disc.step()
                
                # Discriminator accuracy
                pred_cls = pred_concat.detach().gt(0.5).float().squeeze().cpu().numpy()
                disc_pred.append(pred_cls)
                disc_labels.append(label_concat.detach().squeeze().cpu().numpy())

                #######################
                # train target model  #
                #######################

                optimizer_encoder.zero_grad()
                optimizer_disc.zero_grad()

                pred_tgt_strong = self.feature_discriminator(target_feat_strong.detach())
                pred_tgt_weak = self.feature_discriminator(target_feat_weak.detach())
                pred_tgt= torch.cat((pred_tgt_strong,pred_tgt_weak),dim=0)

                # prepare fake labels
                label_tgt = (torch.ones(pred_tgt_strong.size(0)*2)).to(self.device)
                loss_adv = criterion_disc(pred_tgt.squeeze(), label_tgt.float())

                # perform random logit interpolation
                source_pred = torch.cat((source_pred_strong,source_pred_weak),dim = 0)
                pred_source_tgtmodel = torch.cat((pred_source_tgtmodel_strong,pred_source_tgtmodel_weak),dim = 0)
                lambd = torch.rand_like(source_pred,device=self.device)
                final_logits_source = (lambd * source_pred) + ((1-lambd) * pred_source_tgtmodel)
                
                logits_source_weak = final_logits_source[int(source_data_weak.size(0)//2):]
                # pseudolabels_source = F.softmax(logits_source_weak, 1)
                pseudolabels_source = logits_source_weak

                ## softmax for logits of weakly augmented target data
                # pseudolabels_target = F.softmax(pred_target_weak, 1)
                pseudolabels_target = pred_target_weak


                ## allign target label distribtion to source label distribution
               
                # perform relative confidence thresholding
                row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
                final_sum = torch.mean(row_wise_max, 0)

                ## define relative confidence threshold
                c_tau = tau * final_sum

                max_values, _ = torch.max(pseudolabels_target, dim=1)
                mask = (max_values >= c_tau).float()

                pseudolabels_target = torch.max(pseudolabels_target, 1)[1].detach() # argmax

                # compute loss
                source_labels = torch.cat((source_strong_labels,source_weak_labels),dim = 0)
                source_loss_label = self._compute_source_loss(final_logits_source,source_labels)
                target_loss_label = self._compute_target_loss(pseudolabels_target, pred_target_strong, mask,self.device)

                # ## compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / self.epochs_target)) / 2

                ## get total loss
                # self.logger.debug(f'loss_tgt: {loss_tgt}; target_loss_label: {target_loss_label}; source_loss_label:{source_loss_label}')
                loss_ada =  source_t * source_loss_label + (mu * target_loss_label)
                loss =  adv_t*loss_adv + ada_t*loss_ada
                current_step += 1

                # backpropagate and update weights
                loss.backward()
                # optimize target encoder
                optimizer_encoder.step()
                optimizer_disc.step()
                # metrics
                running_loss += loss.item()



            target_loss = running_loss / iters

            disc_pred = np.concatenate(disc_pred)
            disc_labels = np.concatenate(disc_labels)
            acc_disc = sklearn.metrics.accuracy_score(disc_labels, disc_pred)

            self.history_target['acc_disc'].append(acc_disc)
            self.history_target['target_loss'].append(target_loss_label.item())
            # self.evaluate on testing data (target domain)\

            epoch_accuracy_target,labels_list_target,_,preds_list_target = self.evaluate(self.target_model,target_dataloader_weak,return_lists_roc=True)

            torch.set_printoptions(profile="full")
            self.history_target['accuracy_target'].append(epoch_accuracy_target)

            # save checkpoint
            if epoch_accuracy_target > best_acc:
                torch.save({'target_model_weights': self.target_model.state_dict(),
                             },os.path.join(save_path,'model_save','ckpt_targetmodel_discriminator.pt'))
                best_acc = epoch_accuracy_target
                bad_epochs = 0
                
            else:
                bad_epochs += 1
                
            self.logger.debug(f'[Epoch {epoch+1}/{self.epochs_target}] acc_disc: {acc_disc}; target_loss: {target_loss_label:.6f}; accuracy target: {epoch_accuracy_target:.6f}')
            writer_func.save_scalar(self.writer, {'Loss/epoch':target_loss, 'Accuracy/target':epoch_accuracy_target}, epoch)
            
            if bad_epochs >= patience:
                self.logger.debug(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

            # scheduler step
            if step_scheduler:
                scheduler.step()

        best = torch.load(os.path.join(save_path,'model_save','ckpt_targetmodel_discriminator.pt'))
        writer_func.save_text(self.writer, {'best_acc':str(best_acc)})
        self.target_model.load_state_dict(best['target_model_weights'])
        # self.feature_discriminator.load_state_dict(best['discriminator_weights'])

        np.save(his_path_target,self.history_target)

        return self.target_model#, self.feature_discriminator

    def Trainer(self,source_train_dataloader_srtong,source_train_dataloader_weak,source_valid_dataloader_strong,source_valid_dataloader_weak,source_test_dataloader_strong,source_test_dataloader_weak, hyperparams, save_path, his_path):
        # configure hyperparameters
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        step_scheduler = hyperparams['step_scheduler']

        self.logger.debug(f"learning rate: {lr}")
        self.logger.debug(f"weight_decay: {wd}")
        self.logger.debug(f"step_scheduler: {step_scheduler}")

        # source_train_dataloader_srtong,source_train_dataloader_weak = zip(*source_train_dataloader)
        iters = max(len(source_train_dataloader_srtong),len(source_valid_dataloader_strong),len(source_test_dataloader_strong))
        # source_train_dataloader = zip(source_train_dataloader_srtong,source_train_dataloader_weak)

        loss_function = nn.CrossEntropyLoss().to(self.device)
        # configure optimizer and scheduler
        optimizer = optim.Adam(list(self.source_model.parameters()), lr=lr, weight_decay=wd)

        if step_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 500
        bad_epochs = 0
        

        self.history_source = {'source_loss': [], 'accuracy_source_train': [], 'accuracy_source_valid': [], 'accuracy_source_test':[]}
        
        self.source_model.train() 
        # training loop
        for epoch in range(start_epoch, self.epochs_source):
            running_loss = 0.0
            for (strong_data_train, strong_labels_train),(weak_data_train,weak_labels_train) in zip(source_train_dataloader_srtong,source_train_dataloader_weak):
                strong_data_train = strong_data_train.type(torch.FloatTensor).to(self.device)
                strong_labels_train = strong_labels_train.type(torch.LongTensor).to(self.device)
                weak_data_train = weak_data_train.type(torch.FloatTensor).to(self.device)
                weak_labels_train = weak_labels_train.type(torch.LongTensor).to(self.device)


                data_source_train = torch.cat((strong_data_train,weak_data_train),dim=0)
                labels_source_train = torch.cat((strong_labels_train,weak_labels_train),dim=0)

                # forward pass: calls the model once for both source and target and once for source only
                pred,(_,_) = self.source_model(data_source_train)
            
                loss = loss_function(pred, labels_source_train)
                
                # zero gradients
                optimizer.zero_grad()
                # backpropagate and update weights

                loss.backward()
                optimizer.step()

                # metrics
                running_loss += loss.item()

            # get losses
          
            source_loss = running_loss / iters
            # source_loss = source_loss
            #  / iters
            self.history_source['source_loss'].append(source_loss)
            # print('source_train_dataloader',source_train_dataloader)
            
            accuracy_source_train, _, _, _ = self.evaluate_two(self.source_model,zip(source_train_dataloader_srtong,source_train_dataloader_weak),return_lists_roc=True)
            accuracy_source_valid, _, _, _ = self.evaluate_two(self.source_model,zip(source_valid_dataloader_strong,source_valid_dataloader_weak),return_lists_roc=True)

            self.history_source['accuracy_source_train'].append(accuracy_source_train)
            self.history_source['accuracy_source_valid'].append(accuracy_source_valid)

            # save checkpoint
            if accuracy_source_valid > best_acc:
                torch.save({'source_model_weights': self.source_model.state_dict()
                            }, save_path)
                best_acc = accuracy_source_valid
                
                bad_epochs = 0
            else:
                bad_epochs += 1
                

            self.logger.debug('[Epoch {}/{}] source_train_loss: {:.6f}; accuracy source train: {:.6f}; accuracy source valid: {:.6f}'.format(epoch+1, self.epochs_source, source_loss, accuracy_source_train, accuracy_source_valid))
            writer_func.save_scalar(self.writer, {'Loss/source_train':source_loss,'Accuracy/source_train':accuracy_source_train, 'Accuracy/source_valid':accuracy_source_valid}, epoch)
            
            if bad_epochs >= patience:
                self.logger.debug(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

            # scheduler step
            if step_scheduler:
                scheduler.step()

        writer_func.save_text(self.writer, {'best_acc_valid_source':str(best_acc)})
        self.logger.debug(f'best_acc_valid_source: {best_acc}')
        best = torch.load(save_path)
        self.source_model.load_state_dict(best['source_model_weights'])

        accuracy_target, labels_list_test, outputs_list_test, preds_list_test = self.evaluate_two(self.source_model,zip(source_test_dataloader_strong,source_test_dataloader_weak),return_lists_roc=True)
        writer_func.save_text(self.writer, {'acc_test':str(accuracy_target)})
        # print("target_accuracy:",accuracy_target)
        self.logger.debug(f'acc_test_source : {accuracy_target}')
        np.save(his_path,self.history_source)

        return self.source_model

    def evaluate(self, model, dataloader, return_lists_roc=False):
        # set network to evaluation mode
        model.eval()
        # self.model.eval()
        # size=len(dataloader[0].dataset)

        labels_list = []
        outputs_list = []
        preds_list = []
        # print('model.parameters()',next(model.parameters()))
        with torch.no_grad():
            
            # for (data_strong, labels_strong),(data_weak, labels_weak) in dataloader:
            for data, labels in dataloader:
                # print('data',data)
                # self.logger.debug(f"data target:{data[0][0]}")
                # data = data.to(self.device)
                # labels = labels.to(self.device)
                # self.logger.debug(f"data target:{data[0][0]}")
                # data = torch.from_numpy(np.concatenate((data_strong,data_weak),axis=0))
                # labels = torch.from_numpy(np.append(labels_strong,labels_weak))

                data = data.to(self.device)
                labels = labels.to(self.device)
                # print(f"mean: {data.mean()} std: {data.std()}")

                # predict
                logits,(_,_) = model(data)
                # outputs = F.softmax(self.model(data), dim=1)

                # numpify
                labels_numpy = labels.detach().cpu().numpy()
                outputs_numpy = logits.detach().cpu().numpy() # probs (AUROC)
                
                preds = np.argmax(outputs_numpy, axis=1) # accuracy

                # append
                labels_list.append(labels_numpy)
                outputs_list.append(outputs_numpy)
                preds_list.append(preds)
            
            labels_list = np.concatenate(labels_list)
            # self.logger.debug(f'labels_list : {labels_list}')
            self.logger.debug(f"labels_list pred:  {Counter(labels_list.tolist())}")
            outputs_list = np.concatenate(outputs_list)
            preds_list = np.concatenate(preds_list)
            self.logger.debug(f"preds_list pred:  {Counter(preds_list.tolist())}")
            # self.logger.debug(f'preds_list : {preds_list}')

        # metrics
        # auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        # print(labels_list[0],preds_list[0])
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list
            
        return accuracy

    def evaluate_two(self, model, dataloader, return_lists_roc=False):
        # set network to evaluation mode
        model.eval()
        # self.model.eval()

        labels_list = []
        outputs_list = []
        preds_list = []

        with torch.no_grad():
            for ((strong_data, strong_labels),(weak_data,weak_labels)) in dataloader:
                # print('strong_data,strong_labels',strong_data,strong_labels)
                strong_data = strong_data.to(self.device)
                strong_labels = strong_labels.to(self.device)
                weak_data = weak_data.to(self.device)
                weak_labels = weak_labels.to(self.device)

                data=torch.cat((strong_data,weak_data),dim=0)
                labels=torch.cat((strong_labels,weak_labels),dim=0)
                # print(data.mean())

                # predict
                logits,(_,_) = model(data)

                # numpify
                labels_numpy = labels.detach().cpu().numpy()
                outputs_numpy = logits.detach().cpu().numpy() # probs (AUROC)
                
                preds = np.argmax(outputs_numpy, axis=1) # accuracy

                # append
                labels_list.append(labels_numpy)
                outputs_list.append(outputs_numpy)
                preds_list.append(preds)
            # print('labels_list',labels_list)
            
            labels_list = np.concatenate(labels_list)
            outputs_list = np.concatenate(outputs_list)
            self.logger.debug(f"labels_list pred:  {Counter(labels_list.tolist())}")
            preds_list = np.concatenate(preds_list)
            self.logger.debug(f"preds_list pred:  {Counter(preds_list.tolist())}")

        # metrics
        # auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        # print(labels_list[0],preds_list[0])
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list
            
        return accuracy

    def plot_metrics(self,savepath,now):
        """
        Plots the training metrics (only usable after calling .train()).
        """
        savepath = os.path.join(savepath,"pictures")
        savepath =get_nonexistant_path(savepath)
        os.makedirs(savepath)

        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 5, figsize=(18,5), dpi=200)

        epochs_source = len(self.history_source['source_loss'])
        epochs_target = len(self.history_target['target_loss'])

        axs[0].plot(range(1, epochs_source+1), self.history_source['source_loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Entropy loss')

        axs[1].plot(range(1, epochs_source+1), self.history_source['accuracy_source_train'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy on source train')

        axs[2].plot(range(1, epochs_source+1), self.history_source['accuracy_source_valid'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].set_title('Accuracy on source valid')

        axs[3].plot(range(1, epochs_target+1), self.history_target['target_loss'])
        axs[3].set_xlabel('Epochs')
        axs[3].set_ylabel('Loss')
        axs[3].set_title('loss on target')

        axs[4].plot(range(1, epochs_target+1), self.history_target['accuracy_target'])
        axs[4].set_xlabel('Epochs')
        axs[4].set_ylabel('Accuracy')
        axs[4].set_title('Accuracy on weakly augmented target')      
            
        plt.show()
        plt.savefig(f'{savepath}/metrics_{now}.png')
        return savepath

    def plot_cm_roc(self, dataloader,savepath ,now,n_classes=5):
        """
        Plots the confusion matrix and ROC curves of the model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoadert
            DataLoader with test data.

        n_classes: int
            Number of classes.
        """

        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        self.source_model.eval()
        self.target_model.eval()
        # self.model.eval()

        accuracy, labels_list, outputs_list, preds_list = self.evaluate(self.target_model, dataloader, return_lists_roc=True)

        # plot confusion matrix
        cm = sklearn.metrics.confusion_matrix(labels_list, preds_list)
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(n_classes,n_classes)
        #tn, fp, fn, tp = cm.ravel()

        plt.figure(figsize=(10,10), dpi=200)
        sns.heatmap(cm, annot=labels, cmap=cmap, fmt="")
        plt.title("Confusion matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.show()

        # plot roc
        ## one hot encode data
        onehot = np.zeros((labels_list.size, labels_list.max()+1))
        onehot[np.arange(labels_list.size),labels_list] = 1
        onehot = onehot.astype('int')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        ## get roc curve and auroc for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(onehot[:, i], outputs_list[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        ## get macro average auroc
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(9,9), dpi=200)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"AUC class {i} = {roc_auc[i]:.4f}")

        plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average AUC = {roc_auc['macro']:.4f}", color='deeppink', linewidth=2)
            
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(f'{savepath}/result_{now}.png')

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss() # default: `reduction="mean"`
        loss = loss_function(logits, labels)

        #return weak_loss + strong_loss
        return loss

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask, device):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.80, 1.0, 1.25, 1.20]).to(device))
        pseudolabels = pseudolabels.detach() # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)
        
        return (loss * mask).mean()
        # return loss.mean()