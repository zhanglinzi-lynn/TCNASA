import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F





## CausalConv1d Block

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))




## Dilated-casual CNN Architecture

class DilatedCNN(nn.Module):
    
    def __init__(self,input_channels,output_channels,kernel_size):
        super(DilatedCNN, self).__init__()

        self.dil1 = CausalConv1d(in_channels=input_channels,out_channels=32,kernel_size=kernel_size,
                                dilation=1)
        self.dil2 = CausalConv1d(in_channels=32,out_channels=16,kernel_size=kernel_size,
                                dilation=2)
        self.dil3 = CausalConv1d(in_channels=16,out_channels=output_channels,kernel_size=kernel_size,
                                dilation=4)
        

    def forward(self, x):
        

        x = torch.relu(self.dil1(x))
        x = torch.relu(self.dil2(x))
        x = torch.relu(self.dil3(x))
    


        return x
        




class context_embedding(torch.nn.Module):
    def __init__(self,input_channels=1,embedding_size=256,kernel_size=5):
        super(context_embedding,self).__init__()
        self.dilated_causal_convolution = DilatedCNN(input_channels,embedding_size,kernel_size)

    def forward(self,x):
        x = self.dilated_causal_convolution(x)
        return F.sigmoid(x)







