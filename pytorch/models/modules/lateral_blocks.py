import torch
import torch.nn as nn

class BasicLatBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicLatBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv(x) 