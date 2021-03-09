'''
TODO
@Description:注意力机制
@Author:Zigar
@Date:2021/03/05 16:31:19
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------------------------------------------------------#
#    TODO
#    SEnet_Module
#---------------------------------------------------------------#
class SEModule(nn.Module):
    def __init__(self):
        super(SEModule, self).__init__()

    def forward(self, x):
        return x


#---------------------------------------------------------------#
#    TODO
#    CBAM_module
#---------------------------------------------------------------#
class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()

    def forward(self, x):
        return x