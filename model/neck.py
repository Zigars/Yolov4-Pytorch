'''
Description: neck 颈部网络SPP+PAN
Author: Zigar
Date: 2021-03-05 10:02:22
'''
from collections import OrderedDict

import torch
import torch.nn as nn
from model.common import Convolutional
from model.CSPDarknet53 import *


#---------------------------------------------------------------#
#    CBL
#    继承Convolution
#    Conv2d + BatchNormalization + LeakyRelu 
#---------------------------------------------------------------#
class Convolutional_leaky(Convolutional):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm = "bn", activate="leaky_relu"):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, norm=norm, activate=activate)



#---------------------------------------------------------------#
#    卷积＋上采样操作
#---------------------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.up_sample = nn.Sequential(
            Convolutional_leaky(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return x


#---------------------------------------------------------------#
#    三次卷积块
#    CBL * 3
#    用于neck网络的SPP结构
#---------------------------------------------------------------#
def make_three_conv(in_channels, out_channels):
    m = nn.Sequential(
        Convolutional_leaky(in_channels, out_channels, 1),
        Convolutional_leaky(out_channels, out_channels * 2, 3),
        Convolutional_leaky(out_channels*2, out_channels, 1),
    )
    return m


#---------------------------------------------------------------#
#    五次卷积块
#    CBL * 5
#    用于neck网络的PANet
#---------------------------------------------------------------#
def make_five_conv(in_channels, out_channels):
    m = nn.Sequential(
        Convolutional_leaky(in_channels, out_channels, 1),
        Convolutional_leaky(out_channels, out_channels * 2, 3),
        Convolutional_leaky(out_channels * 2, out_channels, 1),
        Convolutional_leaky(out_channels, out_channels * 2, 3),
        Convolutional_leaky(out_channels * 2, out_channels, 1)
    )
    return m


#---------------------------------------------------------------#
#    P5 对out5进行处理
#    13,13,1024 -> 13, 13, 512
#    head_cov_1 + SPP结构 + head_cov_2
#    head_cov SPP的输入端和输出端处理
#    SPP结构，利用不同大小的池化核进行池化
#    池化后concat堆叠
#---------------------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes = [5, 9, 13]):
        # self.feature_channels = [64, 128, 256, 512, 1024]
        super(SpatialPyramidPooling, self).__init__()

        self.head_conv_1 = make_three_conv(
            feature_channels[4], feature_channels[3]  # 1024, 512
            )
        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size//2)
                for pool_size in pool_sizes
            ]
        )
        self.head_conv_2 = make_three_conv(
            2 * feature_channels[4], feature_channels[3] # 2048, 512
             )
        self._initialize_weights()

    def forward(self, features):
        # 13, 13, 1024 -> 13, 13, 512
        x = self.head_conv_1(features[2])
        # 13, 13, 512 -> 13, 13, 2048
        out5 = [maxpool(x) for maxpool in self.maxpools[::-1]]
        out5 = torch.cat(out5 + [x], dim=1)
        # 13, 13, 2048 -> 13, 13, 512
        features[2] = self.head_conv_2(out5)

        return features

    #---------------------------------------------------------------#
    #    初始化权重
    #---------------------------------------------------------------#
    def _initialize_weights(self):
        print("**"*10, "initing head_conv and SPP weights", "**"*10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))



#---------------------------------------------------------------#
#    PANet结构
#    在FPN的基础上加入自底向上的结构，
#    将低层的信息传导到高层中，
#    同时减少了高层到低层的信息流通需要穿过的卷积层数。
#---------------------------------------------------------------#
class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()
        # self.feature_channels = [64, 128, 256, 512, 1024]
        
        #---------------------------------------------------------------#
        #    P4 
        #---------------------------------------------------------------#
        self.up_sample_P4 = Upsample(
            feature_channels[3], feature_channels[2]  
            )  # 512, 256
        self.conv_for_P4 = Convolutional_leaky(
            feature_channels[3], feature_channels[2], 1
            )  # 512, 256
        self.make_five_conv_1 = make_five_conv(
            feature_channels[3], feature_channels[2] # 512, 256
        )

        #---------------------------------------------------------------#
        #    P3
        #    out3  52 * 52 * 128
        #---------------------------------------------------------------#
        self.up_sample_P3 = Upsample(
            feature_channels[2], feature_channels[1]  
            )  # 256, 128
        self.conv_for_P3 = Convolutional_leaky(
            feature_channels[2], feature_channels[1], 1
            )  # 256, 128
        self.make_five_conv_2 = make_five_conv(
            feature_channels[2],feature_channels[1]  # 256, 128
        )

        #---------------------------------------------------------------#
        #    P4
        #   out4 26 * 26 * 256
        #---------------------------------------------------------------#
        self.down_sample_P4 = Convolutional_leaky(
            feature_channels[1], feature_channels[2], 3, stride=2  # 128, 256
        )
        self.make_five_conv_3 = make_five_conv(
            feature_channels[3], feature_channels[2] # 512, 256
        )


        #---------------------------------------------------------------#
        #    P5
        #    out5 13 * 13 * 512
        #---------------------------------------------------------------#
        self.down_sample_P5 = Convolutional_leaky(
            feature_channels[2], feature_channels[3], 3, stride=2  # 256, 512
        )
        self.make_five_conv_4 = make_five_conv(
            feature_channels[4], feature_channels[3] # 1024, 512
        )

        self._initialize_weights()

    def forward(self, features):
        #---------------------------------------------------------------#
        #    features为SPP后的features，其中out5是经过了SPP结构的输出
        #    out3:52,52,256  out4:26,26,512  out5:13,13,512
        #---------------------------------------------------------------#
        out3, out4, out5 = features[0], features[1], features[2]

        #---------------------------------------------------------------#
        #    P4
        #    13,13,512 -> 26,26,256
        #---------------------------------------------------------------#
        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.up_sample_P4(out5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(out4)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv_1(P4)

        #---------------------------------------------------------------#
        #    P3：out3
        #    26,26,256 -> 52,52,128
        #---------------------------------------------------------------#
        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.up_sample_P3(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(out3)
        # 52,52,128 + 52,52,128 = 52,52,256
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out3 = self.make_five_conv_2(P3)

        #---------------------------------------------------------------#
        #    P4:out4
        #    52,52,128 -> 26,26,256
        #---------------------------------------------------------------#
        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample_P4(out3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out4 = self.make_five_conv_3(P4)

        #---------------------------------------------------------------#
        #    P5:out5
        #    26,26,256 -> 13,13,512
        #---------------------------------------------------------------#
        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample_P5(out4)
        # 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, out5], axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out5 = self.make_five_conv_4(P5)
        
        return [out3, out4, out5]

    #---------------------------------------------------------------#
    #    初始化权重
    #---------------------------------------------------------------#
    def _initialize_weights(self):
        print("**"*10, "initing PANet weights", "**"*10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


#---------------------------------------------------------------#
#    neck结构
#    SPP+PAN结构
#    52*52*256 -> 52,52,128
#    26*26*512 -> 26,26,256
#    13,13,1024 -> 13,13,512
#---------------------------------------------------------------#
class yolo_neck(nn.Module):
    def __init__(self, feature_channels):
        super(yolo_neck, self).__init__()
        # self.feature_channels = [64, 128, 256, 512, 1024]
        self.spp = SpatialPyramidPooling(feature_channels)
        self.PANet = PANet(feature_channels)

    def forward(self, features):
        features = self.spp(features)
        features = self.PANet(features)

        return features

        


#---------------------------------------------------------------#
#    test model
#---------------------------------------------------------------#
# if __name__ == '__main__':
#     conv_1 = Convolutional(3, 3, 2, 1, activate="leaky_relu")
#     conv_2 = Convolutional_leaky(3, 3, 2, 1)
#     x = torch.randn(1, 3, 416, 416)
#     y1 = conv_1(x)
#     y2 = conv_2(x)
#     print(x.shape)
#     print(y1)
#     print(y2)

if __name__ == "__main__":
    feature_channels = [64, 128, 256, 512, 1024]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(0) if cuda else "cpu")
    backbone = CSPdarknet53(None).to(device)
    # print(backbone)
    neck = yolo_neck(feature_channels).to(device)
    # print(neck)
    x = torch.randn(1, 3, 416, 416).to(device)
    torch.cuda.empty_cache()
    print("*"*8 , "this is backbone output", "*"*8)
    features = backbone(x)
    print(features[0][0].shape)
    print(features[1][0].shape)
    print(features[2][0].shape)
    print("*"*8 , "this is neck output", "*"*8)
    features = neck(features)
    print(features[0][0].shape)
    print(features[1][0].shape)
    print(features[2][0].shape)
        