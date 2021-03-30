'''
@Description:common 公用组件
@Author:Zigar
@Date:2021/03/05 15:37:00
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import SEModule, CBAM
from config.config import cfg

#---------------------------------------------------------------#
#   Mish激活函数
#---------------------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------------------#
#    标准化和激活函数
#---------------------------------------------------------------#
norm_name = {"bn": nn.BatchNorm2d}

activate_name = {
    "relu"      : nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear"    : nn.Identity(),
    "mish"      : Mish(),
}

#---------------------------------------------------------------#
#    Convolutional
#    卷积块 = 卷积 + 标准化 + 激活函数
#    Conv2d + BatchNormalization + Mish / Relu / LeakyRelu 
#---------------------------------------------------------------#
class Convolutional(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels,
        kernel_size,
        stride = 1, 
        norm = "bn",
        activate="mish"):
        super(Convolutional, self).__init__()

        self.norm     = norm
        self.activate = activate

        self.__conv   = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            kernel_size // 2, 
            bias=False
            )
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](out_channels)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky_relu":
                self.__activate = activate_name[activate](
                    negative_slope=0.1, inplace=False
                )
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=False)
            if activate == "mish":
                self.__activate = activate_name[activate]

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        
        return x


#---------------------------------------------------------------#
#    Resblock
#    CSPdarknet结构块的组成部分
#    内部堆叠的小残差块
#    包含注意力机制
#---------------------------------------------------------------#
class Resblock(nn.Module):
    def __init__(
        self, 
        channels,
        hidden_channels = None,
        # residual_activation="linear",
        ):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels
        
        self.block = nn.Sequential(
            Convolutional(channels, hidden_channels, 1),
            Convolutional(hidden_channels, channels, 3)
        )

        # self.activation = activate_name[residual_activation]
        self.attention = cfg.MODEL.ATTENTION["TYPE"]
        if self.attention == "SEnet":
            self.attention_module = SEModule(channels)
        elif self.attention == "CBAM":
            self.attention_module = CBAM(channels)
        else:
            self.attention = None


    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.attention is not None:
            out = self.attention_module(out)
        out += residual
        return out
