'''
@Description:head yolo检测头
@Author:Zigar
@Date:2021/03/05 15:35:53
'''
import torch
import torch.nn as nn
from model.common import Convolutional
from model.CSPDarknet53 import *


class head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(head, self).__init__()

        self.head = nn.Sequential(
            Convolutional(input_channels, 2*input_channels, 3),
            nn.Conv2d(2*input_channels, output_channels, 1)
        )

    def forward(self, x):
        head = self.head(x)

        return head

class yolo_head(nn.Module):
    def __init__(self, output_channels, feature_channels):
        super(yolo_head, self).__init__()

        #---------------------------------------------------------------#
        #    输入为PANet网络的三种尺度的输出
        #    P3: 52*52*128 -> 52,52,255（coco）
        #    P4: 26*26*256 -> 26,26,255（coco）
        #    P5: 13,13,512 -> 13,13,255（coco）
        #---------------------------------------------------------------#
        self.head3 = head(feature_channels[1], output_channels)
        self.head4 = head(feature_channels[2], output_channels)
        self.head5 = head(feature_channels[3], output_channels)

        self._initialize_weights()

    def forward(self, features):
        out3 = self.head3(features[0])
        out4 = self.head4(features[1])
        out5 = self.head5(features[2])

        return [out3, out4, out5]

    def _initialize_weights(self):
        print("**"*10, "initing yolo_head weights", "**"*10)

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



