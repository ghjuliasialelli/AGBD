""" 

This script defines the loss functions used for training and validation.

"""

#######################################################################################################################
# Imports

import torch
import torch.nn as nn

#######################################################################################################################
# Helper functions 

class RMSE(nn.Module):
    """ 
    Root Mean Squared Error.
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.sqrt(torch.mean(weights * self.mse(prediction,target)))


class ME(nn.Module):
    """ 
    Mean Error.
    """

    def __init__(self):
        super(ME, self).__init__()

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * (prediction - target))


class MSE(nn.Module):
    """ 
    Mean Squared Error.
    """

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * self.mse(prediction,target))


class MAE(nn.Module):
    """ 
    Mean Absolute Error.
    """

    def __init__(self):
        super(MAE, self).__init__()
    
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * torch.abs(prediction - target))


class TrainLoss(nn.Module):
    """ 
    Model's training loss.
    """

    def __init__(self, num_outputs, loss_fn):

        super(TrainLoss, self).__init__()
        self.task_num = num_outputs
        
        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        else: raise ValueError('Invalid loss function')

    def forward(self, preds, labels, weights = 1):
        return self.loss_fn(preds, labels, weights)


class ValLoss(nn.Module):
    """ 
    Model's validation loss.
    """

    def __init__(self, num_outputs, loss_fn):

        super(ValLoss, self).__init__()
        self.task_num = num_outputs
        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        else: raise ValueError('Invalid loss function')

    def forward(self, preds, labels):

        losses = {}
        losses['loss'] = self.loss_fn(preds, labels)
        losses['RMSE'] = RMSE()(preds, labels)
        losses['MAE'] = MAE()(preds, labels)
        losses['ME'] = ME()(preds, labels)

        return losses
