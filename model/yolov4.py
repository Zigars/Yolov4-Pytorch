'''
TODO
@Description:yolov4整体结构
@Author:Zigar
@Date:2021/03/05 15:35:24
'''
import torch
import torch.nn as nn
# from model.common import Convolutional
from model.CSPDarknet53 import CSPdarknet53
from model.neck import yolo_neck
from model.head import yolo_head

class Yolo_Body(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(Yolo_Body, self).__init__()
        
        #---------------------------------------------------------------#
        #    输出通道数
        #    输出通道数 = 先验框数量 * （xywh + 置信度 + 检测数量）
        #---------------------------------------------------------------#
        output_channels = num_anchors * (5 + num_classes)

        # 特征通道
        self.feature_channels = [64, 128, 256, 512, 1024]

        #---------------------------------------------------------------#
        #    生成CSPdarknet53的主干模型
        #    输入：图片 
        #    416,416,3
        #    输出：三种尺度的特征图
        #    52,52,256  26,26,512  13,13,1024
        #---------------------------------------------------------------#
        self.backbone = CSPdarknet53(None)

        #---------------------------------------------------------------#
        #    neck 颈部网络SPP+PAN
        #    输入：CSPdarknet的输出，三种尺度的特征图
        #    52*52*256  26*26*512  13,13,1024
        #    输出：经过SPP和PAN的多尺度融合后的三种尺度的特征图
        #    52,52,128  26,26,256  13,13,512
        #---------------------------------------------------------------#
        self.neck = yolo_neck(self.feature_channels)

        #---------------------------------------------------------------#
        #    head yolo检测头
        #    输入：经过SPP和PAN的多尺度融合后的三种尺度的特征图
        #    52,52,128  26,26,256  13,13,512
        #    输出：经过检测头输出的三种尺度的检测结果
        #    52,52,255  26,26,255  13,13,255（coco数据集 255 = 3*(4+1+80)）
        #---------------------------------------------------------------#
        self.head = yolo_head(output_channels, self.feature_channels)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        features = self.head(features)

        return features
