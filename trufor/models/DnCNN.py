# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
DnCNN: Denoising Convolutional Neural Network
Used for Noiseprint++ feature extraction in TruFor.

Created in September 2020
@author: davide.cozzolino
"""

import math
import torch.nn as nn


def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding=None):
    """Create convolution layer with automatic padding calculation."""
    if padding is None:
        padding = kernelsize // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, 
                    dilation=dilation, padding=padding, bias=bias)


def conv_init(conv, act='linear'):
    """Initialize convolution weights (reproduces DnCNN initialization)."""
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))


def batchnorm_init(m, kernelsize=3):
    """Initialize batchnorm weights (reproduces DnCNN initialization)."""
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()


def make_activation(act):
    """Create activation function."""
    if act is None or act == 'linear':
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    else:
        raise ValueError(f"Unknown activation: {act}")


def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum=0.1, padding=None):
    """
    Create a DnCNN-style network.
    
    Args:
        nplanes_in: number of input feature channels
        kernels: list of kernel sizes for convolution layers
        features: list of hidden layer feature channels
        bns: list of whether to add batchnorm layers
        acts: list of activations
        dilats: list of dilation factors
        bn_momentum: momentum of batchnorm
        padding: integer for padding (None for same padding)
    """
    depth = len(features)
    assert len(features) == len(kernels), "Features and kernels must have same length"

    layers = []
    for i in range(depth):
        if i == 0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        # Convolution layer
        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], 
                               dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        # Batch normalization
        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum=bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        # Activation
        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


class DnCNN(nn.Module):
    """
    DnCNN network implementation.
    """
    
    def __init__(self, nplanes_in, nplanes_out, features, kernel, depth, activation, 
                 residual, bn, lastact=None, bn_momentum=0.10, padding=None):
        """
        Initialize DnCNN.
        
        Args:
            nplanes_in: number of input feature channels
            nplanes_out: number of output feature channels
            features: number of hidden layer feature channels
            kernel: kernel size of convolution layers
            depth: number of convolution layers (minimum 2)
            activation: activation function name
            residual: whether to add a residual connection from input to output
            bn: whether to add batchnorm layers
            lastact: activation for last layer
            bn_momentum: momentum of batchnorm
            padding: integer for padding
        """
        super(DnCNN, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in

        kernels = [kernel] * depth
        features = [features] * (depth-1) + [nplanes_out]
        bns = [False] + [bn] * (depth - 2) + [False]
        dilats = [1] * depth
        acts = [activation] * (depth - 1) + [lastact]
        
        self.layers = make_net(nplanes_in, kernels, features, bns, acts, 
                              dilats=dilats, bn_momentum=bn_momentum, padding=padding)

    def forward(self, x):
        """Forward pass through DnCNN."""
        shortcut = x
        x = self.layers(x)

        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:, :nshortcut, :, :] = x[:, :nshortcut, :, :] + shortcut[:, :nshortcut, :, :]

        return x