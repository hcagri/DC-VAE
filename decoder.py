import torch 
import torch.nn as nn 
import torch.nn.functional as F



class ConvBlock(nn.Module):
    ''' Pre-activation Conv Block with no Normalization '''

    def __init__(self, in_channel, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels,kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        return self.conv(F.relu(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        channel_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 4**2 * latent_dim)

        self.conv1 = ConvBlock(latent_dim, channel_dim)
        self.conv2 = ConvBlock(channel_dim, channel_dim)
        self.conv3 = ConvBlock(channel_dim, channel_dim)
        self.conv4 = ConvBlock(channel_dim, channel_dim)
        self.conv5 = ConvBlock(channel_dim, channel_dim)
        self.conv6 = ConvBlock(channel_dim, channel_dim)

        self.to_rgb = nn.Sequential(
            ConvBlock(channel_dim, 3),
            nn.Tanh()
        )

    def forward(self, x):
        
        x = self.linear(x).view(-1, self.latent_dim, 4, 4)
        skip1 = x

        x = nn.Upsample(scale_factor=2,mode='nearest')(x)
        x = self.conv1(x)
        skip2, skip3 = x, x
        x = self.conv2(x)
        x = x + nn.Upsample(scale_factor=2,mode='nearest')(skip1)

        skip4 = x
        x = nn.Upsample(scale_factor=2,mode='bilinear')(x)
        x = self.conv3(x)
        skip5 = x 
        x = x + nn.Upsample(scale_factor=2,mode='bilinear')(skip2)
        x = self.conv4(x)
        x = x + nn.Upsample(scale_factor=2,mode='bilinear')(skip4)

        x = nn.Upsample(scale_factor=2,mode='nearest')(x)
        x = self.conv5(x)
        x = x + nn.Upsample(scale_factor=4,mode='nearest')(skip3) + nn.Upsample(scale_factor=2,mode='nearest')(skip5)

        x = self.conv6(x)

        out = self.to_rgb(x)
        return out



a = torch.randn((10, 128))
model = Decoder(128)

print(model(a).size())

