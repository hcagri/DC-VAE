import torch
from torch import nn 

class ResConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, padding, pre_activation=False, downsample=False):
        super(ResConvBlock, self).__init__()
        self.pre_activation = pre_activation
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding=padding))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(ch_out, ch_out, kernel_size=kernel, padding=padding))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0))
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.ds = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        #residual
        if self.pre_activation:
            hidden = self.relu(x)
        else:
            hidden = x
        hidden = self.relu(self.conv1(x))
        hidden = self.conv2(hidden)
        if self.downsample:
            hidden = self.ds(hidden)
            x = self.ds(x)
        sc = self.shortcut(x)
        return hidden + sc

class Encoder(nn.Module):
    def __init__(self, ch_in, ch_out, z_dim):
        super(Encoder, self).__init__()
        self.block1 = ResConvBlock(ch_in=ch_in, ch_out=ch_out, kernel=3, padding=1, pre_activation=False, downsample=True)
        self.block2 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True, downsample=True)
        self.block3 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True)
        self.block4 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True)
        self.lin = nn.utils.spectral_norm(nn.Linear(ch_out, 2*z_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        hidden = self.relu(self.block4(hidden))
        hidden = hidden.sum(2).sum(2)
        out = self.lin(hidden)
        return out

class Discriminator(nn.Module):
    def __init__(self, ch_in, ch_out, cont_dim):
        super(Discriminator, self).__init__()
        self.block1 = ResConvBlock(ch_in=ch_in, ch_out=ch_out, kernel=3, padding=1, pre_activation=False, downsample=True)
        self.block2 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True, downsample=True)
        self.block3 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True)
        self.block4 = ResConvBlock(ch_in=ch_out, ch_out=ch_out, kernel=3, padding=1, pre_activation=True)
        self.cont_conv = nn.Conv2d(ch_out, 1, kernel_size=1, padding=1)
        self.cont_lin = nn.Linear(81, cont_dim)
        self.flatten = nn.Flatten()
        self.lin = nn.utils.spectral_norm(nn.Linear(ch_out, cont_dim))
        self.disc_lin = nn.utils.spectral_norm(nn.Linear(cont_dim, 1))
        self.relu = nn.ReLU()
    def forward(self, x):
        hidden = self.block1(x)
        hidden = self.block2(hidden)
        cont_hidden = self.flatten(self.cont_conv(hidden))
        cont_out = self.cont_lin(cont_hidden)
        hidden = self.block3(hidden)
        hidden = self.relu(self.block4(hidden))
        hidden = hidden.sum(2).sum(2)
        hidden = self.lin(hidden)
        disc_out = self.disc_lin(hidden)
        return disc_out, cont_out
