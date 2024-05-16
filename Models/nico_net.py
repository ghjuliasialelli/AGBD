"""
A CNN designed for pixel-wise analysis of Sentinel-2 satellite images.
"XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
Any kind of down sampling is avoided (no pooling, striding, etc.).

All details about the architecture are described in:
Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PointwiseBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d):
        super(PointwiseBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = conv1x1(in_channels, filters[0])
        self.bn1 = norm_layer(filters[0])

        self.conv2 = conv1x1(filters[0], filters[1])
        self.bn2 = norm_layer(filters[1])

        self.conv3 = conv1x1(filters[1], filters[2])
        self.bn3 = norm_layer(filters[2])

        self.relu = nn.ReLU(inplace=True)
        self.conv_shortcut = conv1x1(in_channels, filters[2])
        self.bn_shortcut = norm_layer(filters[2])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = self.relu(out)

        return out


class SepConvBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn1 = norm_layer(filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3)
        self.bn2 = norm_layer(filters[1])

        self.relu = nn.ReLU(inplace=False)
        self.conv_shortcut = conv1x1(in_channels, filters[1])
        self.bn_shortcut = norm_layer(filters[1])

    def forward(self, x):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)

        out = self.relu(x)
        out = self.sepconv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.sepconv2(out)
        out = self.bn2(out)

        out = out + shortcut

        return out


class ResLayer(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResLayer, self).__init__()
        self.filters = filters
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=1, bias=True)
        self.w2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y

def ELUplus1(x):
    elu = nn.ELU(inplace=False)(x)
    return torch.add(elu, 1.0)


def clamp_exp(x, min_x=-100, max_x=10):
    x = torch.clamp(x, min=min_x, max=max_x)
    return torch.exp(x)


class XceptionS2(nn.Module):

    def __init__(self, in_channels, out_channels=1, num_sepconv_blocks=8, num_sepconv_filters=728, returns="targets",
                long_skip=False, manual_init=False, freeze_features=False, freeze_last_mean=False):

        super(XceptionS2, self).__init__()

        self.freeze_features = freeze_features
        self.freeze_last_mean = freeze_last_mean  # freeze the last linear regression layers (mean)


        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.returns = returns
        self.long_skip = long_skip

        self.entry_block = PointwiseBlock(in_channels=in_channels, filters=[128, 256, num_sepconv_filters])
        self.sepconv_blocks = self._make_sepconv_blocks()

        self.predictions = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)
        self.second_moments = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)

        # initialize parameters
        if manual_init:
            print('Manual weight init with Kaiming Normal')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') TODO: check if kaiming would be better with ReLU (see torchvision resnet)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)  # gamma
                    nn.init.constant_(m.bias, 0)  # beta

        if self.freeze_features:
            print(
                f'Freezing feature extractor... args.freeze_features={self.freeze_features}'
            )
            # do not train the backbone of the image network
            for param in self.parameters():
                param.requires_grad = False

        # train the last layer(s) of the linear regressor
        if not self.freeze_last_mean:
            print(
                f'Unfreeze last layer (mean regressor)... args.freeze_last_mean={self.freeze_last_mean}'
            )
            for param in self.predictions.parameters():
                param.requires_grad = True

    def _make_sepconv_blocks(self):
        blocks = [
            SepConvBlock(
                in_channels=self.num_sepconv_filters,
                filters=[self.num_sepconv_filters, self.num_sepconv_filters],
            )
            for _ in range(self.num_sepconv_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: input tensor: first 12 channels are sentinel-2 bands, last 3 channels are lat lon encoding
        """

        x = self.entry_block(x)
        if self.long_skip:
            shortcut = x
        x = self.sepconv_blocks(x)
        if self.long_skip:
            x = x + shortcut
        predictions = self.predictions(x)

        if self.returns == "targets":
            return predictions

        else:
            raise ValueError(
                f"XceptionS2 model output is undefined for: returns='{self.returns}'"
            )


# Multi Task Xception ###################################################################################################

class NicoNet(pl.LightningModule) :
    """
        Module defining the Multi Task (MT) version of the Xception architecture. It is made of:
    """

    def __init__(self, in_features, num_outputs):
        """
            - `in_features` (int) : `in_channels` expected by the first layer;
            - `num_outputs` (int) : `out_channels` expected by the last layer of the body;

        """

        super().__init__()

        self.in_features = in_features

        self.intermediary_outputs = num_outputs

        self.body = XceptionS2(in_channels=self.in_features, out_channels=self.intermediary_outputs, num_sepconv_blocks=8, num_sepconv_filters=728, returns="targets",
                long_skip=False, manual_init=False, freeze_features=False, freeze_last_mean=False)


    def forward(self, x) :
        x = self.body(x) # (batch_size, intermediary_outputs, size, size)
        return x
