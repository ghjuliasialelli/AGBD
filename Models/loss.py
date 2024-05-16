""" 

Content: definition of the various loss-related modules and functions used throughout the torch_code/ directory.

Includes:
- `ME()` : (weighted) Mean Error;
- `RMSE()` : (weighted) Root Mean Squared Error;
- `MSE()` : (weighted) Mean Squared Error;
- `CE()` : (weighted) Entropy loss;
- `MAE()` : (weighted) Mean Absolute Error;
- `TrainLoss` : Module for the training loss;
- `ValLoss` : Module for the validation loss.

"""

import torch
import torch.nn as nn

class ME(nn.Module):
    """ 
        Weighted ME.
    """

    def __init__(self):
        super(ME, self).__init__()

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * (prediction - target))

class RMSE(nn.Module):
    """ 
        Weighted RMSE.
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.sqrt(torch.mean(weights * self.mse(prediction,target)))

class MSE(nn.Module):
    """ 
        Weighted MSE.
    """

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * self.mse(prediction,target))

class CE(nn.Module):
    """ 
        Weighted Cross Entropy.
    """

    def __init__(self):
        super(CE, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        target = target.long() - 1
        return torch.mean(weights * self.CE_loss(prediction, target))

class MAE(nn.Module):
    """ 
        Weighted MAE .
    """

    def __init__(self):
        super(MAE, self).__init__()
    
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * torch.abs(prediction - target))

class GaussianNLL(nn.Module):
    """
        Gaussian negative log likelihood to fit the mean and variance to p(y|x)
        Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
        Furthermore, the constant log term is discarded.
    """

    def __init__(self, eps = 1e-6):
        super(GaussianNLL, self).__init__()
        self.eps = eps

    def __call__(self, preds, target, weights = 1):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html 
        """
        prediction, log_variance = preds[:, 0], preds[:, 1]
        stable_variance = torch.exp(log_variance) + self.eps
        loss = 0.5 * (torch.log(stable_variance) + torch.pow(prediction - target, 2) / stable_variance)
        return torch.mean(weights * loss)


class TrainLoss(nn.Module):
    """ 
        Wrapper for the model's training loss.
    """

    def __init__(self, num_outputs, loss_fn):

        super(TrainLoss, self).__init__()
        self.task_num = num_outputs
        
        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        elif loss_fn == 'GNLL' :
            self.loss_fn = GaussianNLL()

    def forward(self, preds, labels, weights = 1):

        return self.loss_fn(preds, labels, weights)


class ValLoss(nn.Module):
    """ 
        Wrapper for the model's validation loss.
    """

    def __init__(self, num_outputs, loss_fn):

        super(ValLoss, self).__init__()
        self.task_num = num_outputs
        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        elif loss_fn == 'GNLL' :
            self.loss_fn = GaussianNLL()

    def forward(self, preds, labels):

        losses = {}
        losses['loss'] = self.loss_fn(preds, labels)
        losses['RMSE'] = RMSE()(preds, labels)
        losses['MAE'] = MAE()(preds, labels)
        losses['ME'] = ME()(preds, labels)

        return losses
