import datetime as dt
import numpy as np

import torch
import torch.nn as nn

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=512,
                             out_channels=512,
                             kernel_size=3)
    def forward(self,x):
        return self.conv(x)

class CompressedConv(nn.Module):
    def __init__(self, cin, cout, r1,r2,r, D=3):
        super(CompressedConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=cin,
                              out_channels=r1,
                              kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=r1,
                              out_channels=r,
                              kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=r,
                              out_channels=r,
                              kernel_size=D)
        self.conv4 = nn.Conv2d(in_channels=r,
                              out_channels=r2,
                              kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=r2,
                              out_channels=cout,
                              kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

def CountSpeed(r1,r2,r, dev="cpu"):
    if dev=="cpu":
        compressed_conv = CompressedConv(512,512,r1,r2,r)
    x = torch.randn((64,512,8,8),)
    if dev=="cuda":
        compressed_conv = CompressedConv(512,512,r1,r2,r).cuda()
        x = x.cuda()
    start = dt.datetime.now()
    cycles = 100
    for i in range(100):
        y = compressed_conv(x)
        y = y +1
    finish = dt.datetime.now()
    total_time = (finish-start).total_seconds()/cycles
    return total_time
    
def CountSpeed_simple(dev="cpu"):
    if dev=="cpu":
        compressed_conv = SimpleConv()
    x = torch.randn((64,512,8,8),)
    if dev=="cuda":
        compressed_conv = SimpleConv().cuda()
        x = x.cuda()
    start = dt.datetime.now()
    cycles = 100
    for i in range(100):
        y = compressed_conv(x)
        y = y +1
    finish = dt.datetime.now()
    total_time = (finish-start).total_seconds()/cycles
    return total_time

time = np.zeros((3,3,6))  
for i,r1 in enumerate((30, 60, 90)):
    for j,r2 in enumerate((30,60, 90)):
        for k,r in enumerate((100, 120, 150, 180, 240, 250)):
            time[i,j,k] = CountSpeed(r1,r2,r)

time_gpu = np.zeros((3,3,6))
for i,r1 in enumerate((30, 60, 90)):
    for j,r2 in enumerate((30,60, 90)):
        for k,r in enumerate((100, 120, 150, 180, 240, 250)):
            time_gpu[i,j,k] = CountSpeed(r1,r2,r, dev='cuda')

np.save("time_cpu", time)
np.save("time_gpu", time_gpu)