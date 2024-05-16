""" 

Definition of the Model() Module, which is the wrapper module for all models.

"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import TrainLoss, ValLoss, RMSE

class Model(pl.LightningModule):

    def __init__(self, model, lr, step_size, gamma, patch_size, downsample, loss_fn):
        """ 
        - `model` (MT_FCN or ResNeXt model) : base model to be trained;
        - `lr` (float) : learning rate;
        - `step_size` (int) : period of learning rate decay;
        - `gamma` (float) : multiplicative factor of learning rate decay;
        - `patch_size` (list) : size of the input patch;
        - `downsample` (bool) : whether to downsample the predictions from 10m to 50m per pixel;
        - `mt_weighting` (str) : multi task losses weighting strategy.
        """ 

        super().__init__()
        self.model = model
        self.num_outputs = model.num_outputs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        
        # With downsampling, we go from 10m per pixel to 50m per pixel
        if downsample:
            self.center = int(patch_size[0] // 5) // 2
        else: 
            self.center = int(patch_size[0] // 2)
        
        self.loss_fn = loss_fn

        self.preds, self.val_preds, self.test_preds = [], [], []
        self.labels, self.val_labels, self.test_labels = [], [], []

        self.TrainLoss = TrainLoss(num_outputs = self.num_outputs, loss_fn = self.loss_fn)
                         
    def training_step(self, batch, batch_idx):

        # split batch
        images, labels = batch
        
        # get prediction
        predictions = self.model(images)
        predictions = predictions[:,:,self.center,self.center]

        # Store the predictions and labels
        self.labels.append(labels)
        self.preds.append(predictions)

        # Return the loss
        loss = self.TrainLoss(predictions, labels)

        return loss

    def on_train_epoch_end(self) :

        # Get the training RMSE and loss
        preds = torch.cat(self.preds).unsqueeze(1)
        labels = torch.cat(self.labels)

        agbd_rmse = RMSE()(preds, labels)
        self.log_dict({'train/agbd_rmse': agbd_rmse, "step": self.current_epoch})

        loss = self.TrainLoss(preds, labels)
        self.log_dict({'train/loss': loss, "step": self.current_epoch})

        # Set the predictions and labels back to empty lists
        self.preds = []
        self.labels = []


    def validation_step(self, batch, batch_idx, dataloader_idx = None):

        # Ordinary validation 
        if dataloader_idx == None or dataloader_idx == 0:
            
            # split batch
            images, labels = batch

            # get predictions
            predictions = self.model(images)
            predictions = predictions[:,:,self.center,self.center]

            # Store the predictions, labels for the on_validation_epoch_end method
            self.val_preds.append(predictions[:, 0])
            self.val_labels.append(labels)
        
        # Validation on the test set
        elif dataloader_idx == 1 :
    
            # split batch
            images, labels = batch

            # get predictions
            predictions = self.model(images)
            predictions = predictions[:,:,self.center,self.center]

            # Store the predictions, labels for the on_validation_epoch_end method
            self.test_preds.append(predictions[:, 0])
            self.test_labels.append(labels)
        
        else: raise ValueError('dataloader_idx should be 0 or 1')
    

    def on_validation_epoch_end(self):
        """
        Calculate the overall validation RMSE and binned metrics.
        """

        # Ordinary validation #####################################################################

        # Log the validation epoch's predictions and labels
        preds = torch.cat(self.val_preds).unsqueeze(1)
        labels = torch.cat(self.val_labels)
        agbd_rmse = RMSE()(preds, labels)
        self.log_dict({'val/agbd_rmse': agbd_rmse, "step": self.current_epoch})

        # Log the validation agbd rmse by bin
        bins = np.arange(0, 501, 50)
        for lb,ub in zip(bins[:-1], bins[1:]):
            pred, label = preds[(lb <= labels) & (labels < ub)], labels[(lb <= labels) & (labels < ub)]
            rmse = RMSE()(pred, label)
            self.log_dict({f'binned/val_rmse_{lb}-{ub}': rmse, "step": self.current_epoch})
        
        # Set the predictions and labels back to empty lists
        self.val_preds = []
        self.val_labels = []

        # Validation on the test set ##############################################################

        # Log the test set agbd rmse
        preds = torch.cat(self.test_preds).unsqueeze(1)
        labels = torch.cat(self.test_labels)
        agbd_rmse = RMSE()(preds, labels)
        self.log_dict({'test/agbd_rmse': agbd_rmse, "step": self.current_epoch})

        # Set the predictions and labels back to empty lists
        self.test_labels = []
        self.test_preds = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)]
