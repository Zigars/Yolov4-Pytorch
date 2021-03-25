'''
TODO
@Description:yolov4训练技巧
@Author:Zigar
@Date:2021/03/11 11:32:42
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image


#---------------------------------------------------------------#
#    计算iou
#---------------------------------------------------------------#