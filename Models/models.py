"""

This file contains the implementation of the different models used in the project.

"""

###################################################################################################
# Imports #########################################################################################
###################################################################################################

import torch
import torch.nn as nn
from nico_net import NicoNet

###################################################################################################
# UNet ############################################################################################
###################################################################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None, leaky_relu=False, stride=1):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.relu = nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            self.relu
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            self.relu
        )

        # Adjusting for size and channel discrepancy in the residual connection
        self.match_channels = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.match_channels(x)
        h = self.conv2(self.conv1(x))
        return self.relu(h + residual)

class Down(nn.Module):
    """Double conv"""

    def __init__(self, in_channels, out_channels, leaky_relu=False):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, leaky_relu=leaky_relu,stride=2)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,  leaky_relu=False,odd = True):
        super().__init__()

        if odd: self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1)
        else: self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)

        self.conv = DoubleConv(in_channels, out_channels, leaky_relu=leaky_relu)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def get_layers(patch_size, leaky_relu, basis):
    """
    Get the layers for the UNet model.

    Args:
    - patch_size: list of two integers, the size of the patches.
    - leaky_relu: bool, whether to use leaky ReLU activation functions.
    - basis: list of integers, the dimensions for the UNet model.

    Returns:
    - down_layers: list of Down layers.
    - up_layers: list of Up layers.
    """

    lbs, ubs = basis[:-1], basis[1:]
    assert len(lbs) == len(ubs), 'The number of down and up layers must be the same.'
    
    down_layers, up_layers = [], []
    dim = patch_size[0]
    for lb, ub in zip(lbs, ubs):

        print(f'Down({lb}, {ub})', dim)
        down_layers.append(Down(lb, ub, leaky_relu))

        if dim % 2 == 0:
            odd = False
            dim = (dim // 2)
        else:
            odd = True
            dim = (dim // 2) + 1

        print(f'Up({ub}, {lb})')
        up_layers.append(Up(ub, lb, leaky_relu, odd = odd))

    up_layers.reverse()
    return down_layers, up_layers


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, patch_size, leaky_relu = False):
        """
        A simple UNet implementation.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Set the architecture of the UNet model based on the patch size, such
        # that the number of parameters is approximately 10M in both cases
        assert patch_size[0] == patch_size[1], 'Only square patches are supported.'
        if patch_size[0] == 25 :
            basis = [64, 128, 256, 512]
        elif patch_size[0] == 15 :
            basis = [32, 64, 128, 256, 512]
        else: raise ValueError('Patch_size should be 15 or 25 for UNet architecture.')

        self.inc = DoubleConv(n_channels, basis[0], leaky_relu = leaky_relu)
        down_layers, uplayers = get_layers(patch_size, leaky_relu, basis)
        self.down_layers = nn.ModuleList(down_layers)
        self.uplayers = nn.ModuleList(uplayers)
        self.outc = nn.Conv2d(basis[0], n_classes, kernel_size=1)

    def forward(self, x):

        x = self.inc(x)

        encoder_outputs = [x]
        for layer in self.down_layers:
            x = layer(encoder_outputs[-1])
            encoder_outputs.append(x)

        x = encoder_outputs.pop(-1)
        for layer in self.uplayers:
            prev = encoder_outputs.pop(-1)
            x = layer(x, prev)

        logits = self.outc(x)

        return logits



###################################################################################################
# Fully Convolutional Neural Network (FCN) ########################################################
###################################################################################################

class SimpleFCN(nn.Module):
    def __init__(self,
                 in_features=4,
                 channel_dims = (16, 32, 64, 128),
                 num_outputs=1,
                 kernel_size=3,
                 stride=1,
                 max_pool=False,
                 downsample=None):
        """
        A simple fully convolutional neural network.
        """
        super(SimpleFCN, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        layers = list()
        for i in range(len(channel_dims)):
            in_channels = in_features if i == 0 else channel_dims[i-1]
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=channel_dims[i], 
                                    kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(num_features=channel_dims[i]))
            layers.append(self.relu)
            
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

        if downsample=="max":
            layers.append(nn.MaxPool2d(kernel_size=5, stride=5, padding=0))
        elif downsample=="average":
            layers.append(nn.AvgPool2d(kernel_size=5, stride=5, padding=0))

        self.conv_layers = nn.Sequential(*layers)
        
        self.conv_output = nn.Conv2d(in_channels=channel_dims[-1], out_channels=num_outputs, kernel_size=1,
                                     stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        predictions = self.conv_output(x)
        return predictions


###################################################################################################
# Wrapper #########################################################################################
###################################################################################################

class Net(nn.Module):
    """
    This class is a wrapper around the different models.
    """
    def __init__(self, model_name, in_features = 4, num_outputs = 1, channel_dims = (16, 32, 64, 128), 
                 max_pool = False, downsample = None, leaky_relu = False, patch_size = [15, 15], 
                 compile = True, num_sepconv_blocks = 8, num_sepconv_filters = 728, long_skip = False):
        super(Net, self).__init__()
        
        self.model_name = model_name
        self.num_outputs = num_outputs
        
        # FCN
        if self.model_name == 'fcn' :
            self.model = SimpleFCN(in_features, channel_dims, num_outputs = 1, max_pool = max_pool, 
                                   downsample = downsample)

        # UNet
        elif self.model_name == 'unet' :
            self.model = UNet(n_channels = in_features, n_classes = num_outputs, patch_size = patch_size, 
                               leaky_relu = leaky_relu)
        
        # Nico's model
        elif self.model_name == "nico":
            if compile: self.model = torch.compile(NicoNet(in_features = in_features, num_outputs = num_outputs, 
                                        num_sepconv_blocks = num_sepconv_blocks, 
                                        num_sepconv_filters = num_sepconv_filters, 
                                        long_skip = long_skip))
            else: self.model = NicoNet(in_features = in_features, num_outputs = num_outputs, 
                                        num_sepconv_blocks = num_sepconv_blocks, 
                                        num_sepconv_filters = num_sepconv_filters, 
                                        long_skip = long_skip)

        else:
            raise NotImplementedError(f'unknown model name {model_name}')
        
    def forward(self, x):
        return self.model(x)
