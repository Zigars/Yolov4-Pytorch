'''
@Description:utils工具包
@Author:Zigar
@Date:2021/03/11 11:35:40
'''
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

#---------------------------------------------------------------#
#    计算先验框和真实框之间的交并比iou
#    https://blog.csdn.net/laizi_laizi/article/details/103463802
#    https://wstchhwp.blog.csdn.net/article/details/108450062
#---------------------------------------------------------------#
def box_iou(_box_a, _box_b):
    """
    @param:
    -------
    box_a:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    box_b:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    @Returns:
    -------
    iou:tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # box_a, box_b左上角(x1, y1)和右下角坐标(x2, y2)
    b1_x1, b1_y1 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 1] - _box_a[:, 3] / 2
    b1_x2, b1_y2 = _box_a[:, 0] + _box_a[:, 2] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # box_b左上角和右下角坐标
    b2_x1, b2_y1 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 1] - _box_b[:, 3] / 2
    b2_x2, b2_y2 = _box_b[:, 0] + _box_b[:, 2] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    # 生成两个和所给框相同shape的全0数组
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    # 保存转换后的框坐标(x,y,w,h) -> (x1,y1,x2,y2)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    # 计算交集坐标
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_a[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # 计算交集面积
    inter_hw = torch.clamp((max_xy - min_xy), min=0)
    inter = inter_hw[:, :, 0] * inter_hw[:, :, 1]
    # print(inter)

    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]

    # 计算并集面积
    union = area_a + area_b - inter
    # print(union)

    # 计算IOU
    IOU = inter / union  # [A, B]

    return IOU

# # test iou
# _box_a = torch.from_numpy(np.array([[100, 50, 200, 100]])).float()
# _box_b = torch.from_numpy(np.array([[100, 100,200, 200]])).float()
# print(box_iou(_box_a, _box_b))

#---------------------------------------------------------------#
#    对真实标签做平滑处理
#    https://blog.csdn.net/neveer/article/details/91646657
#---------------------------------------------------------------#
def smooth_labels(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


#---------------------------------------------------------------#
#    ciou
#    https://blog.csdn.net/weixin_43593330/article/details/108450996
#    https://blog.csdn.net/neil3611244/article/details/113794197
#    clamp():
#    输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
#    https://blog.csdn.net/u013230189/article/details/82627375
#---------------------------------------------------------------#
def box_ciou(_box_a, _box_b):
    """
    @param:
    -------
    box_a:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    box_b:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    @Returns:
    -------
    ciou:tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角坐标（x1，y1）右下角坐标(x2，y2)
    b1_xy = _box_a[:, 0:2]
    b1_wh = _box_a[:, 2:4]
    b1_min = b1_xy - b1_wh / 2.
    b1_max = b1_xy + b1_wh / 2.
    # 求出预测框左上角坐标（x1，y1）右下角坐标(x2，y2)
    b2_xy = _box_b[:, 0:2]
    b2_wh = _box_b[:, 2:4]
    b2_min = b2_xy - b2_wh / 2.
    b2_max = b2_xy + b2_wh / 2. 

    # 计算iou
    iou = box_iou(_box_a, _box_b)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_min = torch.min(b1_min, b2_min)
    enclose_max = torch.max(b1_max, b2_max)
    enclose_wh = torch.max(enclose_max - enclose_min, torch.zeros_like(b1_max))
    
    # 计算对角线的距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)

    # 计算diou
    diou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    # 计算ciou
    # 宽高比例系数
    ratio_1 = torch.atan(b1_wh[:, 0] / torch.clamp(b1_wh[:, 1], min=1e-6))
    ratio_2 = torch.atan(b2_wh[:, 0] / torch.clamp(b2_wh[:, 1], min=1e-6))
    v = (4 / (math.pi ** 2)) * torch.pow((ratio_1 - ratio_2), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = diou - alpha * v

    return ciou

# # test ciou
# _box_a = torch.from_numpy(np.array([[100, 50, 200, 100]])).float()
# _box_b = torch.from_numpy(np.array([[100, 100,200, 200]])).float()
# print(box_ciou(_box_a, _box_b))


#---------------------------------------------------------------#
#    防止梯度爆炸，对应tf.clip_by_value()
#    https://blog.csdn.net/york1996/article/details/89434935
#---------------------------------------------------------------#
def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * t + (result > t_max) * t_max
    return result

# # test clip_by_tensor
# b=torch.randint(0,10,(3,3))
# print(b)
# min=torch.randint(0,5,(3,3))
# print(min)
# max=torch.randint(6,10,(3,3))
# print(max)
# print(clip_by_tensor(b,min,max))
# # print(torch.clamp(b, min , max))
# print(torch.Tensor.clip(b, min, max))