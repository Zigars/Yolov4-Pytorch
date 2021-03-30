'''
@Description:utils工具包
@Author:Zigar
@Date:2021/03/11 11:35:40
'''
from numpy.core.defchararray import encode
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.ops import nms
from PIL import Image
import cv2

#---------------------------------------------------------------#
#    解码输出检测框
#---------------------------------------------------------------#
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors     = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs  = 5 + num_classes
        self.img_size    = img_size

    def forward(self, input):
        #---------------------------------------------------------------#
        #    输入的input一共有三层，shape分别是：
        #    batch_size, 255, 13, 13
        #    batch_size, 255, 26, 26
        #    batch_size, 255, 52, 52
        #---------------------------------------------------------------#
        batch_size = input.size(0)
        input_h    = input.size(2)
        input_w    = input.size(3)

        #---------------------------------------------------------------#
        #    输入为416*416时
        #    stride_h = stride_w = 32, 16, 8
        #---------------------------------------------------------------#
        stride_h = self.img_size[1] / input_h
        stride_w = self.img_size[0] / input_w

        # 获得相对于特征层的anchors
        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in self.anchors]

        #---------------------------------------------------------------#
        #    转换input的shape为
        #    batch_size, 3, 13, 13, 85
        #    batch_size, 3, 26, 26, 85
        #    batch_size, 3, 52, 52, 85
        #---------------------------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_h, input_w).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]

        # 获得置信度，是否有目标
        conf = torch.sigmoid(prediction[..., 4])

        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #---------------------------------------------------------------#
        #    生成网格，先验框中心，网格左上角
        #    torch.linspace():返回一个一维的tensor，
        #    这个张量包含了从start到end，分成steps个线段得到的向量
        #    https://blog.csdn.net/york1996/article/details/81671128
        #    repeat():沿着指定的维度重复tensor
        #---------------------------------------------------------------#
        grid_x = torch.linspace(0, input_w - 1, input_w).repeat(input_h, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_h - 1, input_h).repeat(input_w, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 按照网格格式生成先验框的宽高
        # batch_size, 3, 13, 13
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(h.shape)

        #---------------------------------------------------------------#
        #    利用预测结果对先验框进行调整
        #    首先调整先验框的中心，从先验框中心向右下角偏移
        #    再调整先验框的宽高
        #---------------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # self.show_anchor(input_h, pred_boxes, grid_x, grid_y, anchor_w, anchor_h)

        # 将输出调整为相对于416*416的大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data

    #---------------------------------------------------------------#
    #    绘图显示先验框调整过程
    #---------------------------------------------------------------#
    def show_anchor(self, input_height, pred_boxes, grid_x, grid_y, anchor_w, anchor_h):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        if input_height==13:
            plt.ylim(0,13)
            plt.xlim(0,13)
        elif input_height==26:
            plt.ylim(0,26)
            plt.xlim(0,26)
        elif input_height==52:
            plt.ylim(0,52)
            plt.xlim(0,52)
        plt.scatter(grid_x.cpu(),grid_y.cpu())

        anchor_left = grid_x - anchor_w/2
        anchor_top = grid_y - anchor_h/2

        rect1 = plt.Rectangle([anchor_left[0,0,5,5],anchor_top[0,0,5,5]],anchor_w[0,0,5,5],anchor_h[0,0,5,5],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[0,1,5,5],anchor_top[0,1,5,5]],anchor_w[0,1,5,5],anchor_h[0,1,5,5],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[0,2,5,5],anchor_top[0,2,5,5]],anchor_w[0,2,5,5],anchor_h[0,2,5,5],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        if input_height==13:
            plt.ylim(0,13)
            plt.xlim(0,13)
        elif input_height==26:
            plt.ylim(0,26)
            plt.xlim(0,26)
        elif input_height==52:
            plt.ylim(0,52)
            plt.xlim(0,52)
        plt.scatter(grid_x.cpu(),grid_y.cpu())
        plt.scatter(pred_boxes[0,:,5,5,0].cpu(),pred_boxes[0,:,5,5,1].cpu(),c='r')

        pre_left = pred_boxes[...,0] - pred_boxes[...,2]/2
        pre_top = pred_boxes[...,1] - pred_boxes[...,3]/2

        rect1 = plt.Rectangle([pre_left[0,0,5,5],pre_top[0,0,5,5]],pred_boxes[0,0,5,5,2],pred_boxes[0,0,5,5,3],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0,1,5,5],pre_top[0,1,5,5]],pred_boxes[0,1,5,5,2],pred_boxes[0,1,5,5,3],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0,2,5,5],pre_top[0,2,5,5]],pred_boxes[0,2,5,5,2],pred_boxes[0,2,5,5,3],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()


#---------------------------------------------------------------#
#    给图像增加灰条，实现不失真的resize
#    这里bubbliiiing把letterbox_img用在检测的部分
#    我觉得这个方法如果放在训练中应该会有提升
#    或者说这本来就是一种训练的数据增强方法
#    https://blog.csdn.net/qq_36767550/article/details/112499530
#---------------------------------------------------------------#
def letterbox_image(image, size):
    """
    @param:
    -------
    image:输入的图片
    size：resize的大小
    @Returns:
    -------
    new_image:增加灰条后的图片
    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new('RGB', size, (128, 128, 128))
    new_img.paste(image, ((w - nw)//2, (h - nh)//2))
    return new_img


#---------------------------------------------------------------#
#    修正使用letterbox_img的图片的框
#    这段代码没有注释 无法理解
#    具体就是把boxs也按照scale放大后，根据黑边移动相应的距离
#---------------------------------------------------------------#
def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    """
    @param:
    -------
    top, left, bottom, right：letter_img图片检测结果的上下左右距离
    input_shape：输入图片的尺寸，这里是resize后的图片（416，416）
    image_shape：要输出的原始图片尺寸
    @Returns:
    -------
    boxs：修正后的检测框集合
    """
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. 
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxs = box_yx + (box_hw / 2.)

    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxs[:, 0:1],
        box_maxs[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


#---------------------------------------------------------------#
#    计算先验框和真实框之间的交并比iou
#    https://blog.csdn.net/laizi_laizi/article/details/103463802
#    https://wstchhwp.blog.csdn.net/article/details/108450062
#---------------------------------------------------------------#
def box_iou(_box_a, _box_b, x1y1x2y2=False):
    """
    @param:
    -------
    box_a:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    box_b:tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    @Returns:
    -------
    iou:tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    if not x1y1x2y2:
        # box_a, box_b左上角(x1, y1)和右下角坐标(x2, y2)
        b1_x1, b1_y1 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 1] - _box_a[:, 3] / 2
        b1_x2, b1_y2 = _box_a[:, 0] + _box_a[:, 2] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
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
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
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
        iou = inter / union  # [A, B]
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = _box_a[:, 0], _box_a[:, 1], _box_a[:, 2], _box_a[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = _box_b[:, 0], _box_b[:, 1], _box_b[:, 2], _box_b[:, 3]

        # print(b1_x1, b1_y1, b1_x2, b1_y2)
        # print(b2_x1, b2_y1, b2_x2, b2_y2)

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

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


#---------------------------------------------------------------#
#    非极大值抑制nms
#---------------------------------------------------------------#
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    @param:
    -------
    prediction：预测结果框
    num_classes：类别的数量
    conf_thres：置信度阈值
    nms_thres：非极大值抑制阈值
    @Returns:
    output：经过nms的输出框
    -------
    """
    #---------------------------------------------------------------#
    #    xywh -> x1y1x2y2
    #    prediction [batch_size, num_anchors, 85]
    #---------------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    out_put = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #---------------------------------------------------------------#
        #    获取种类预测部分的max
        #    torch.max():返回输入tensor中所有元素的最大值
        #    https://blog.csdn.net/Z_lbj/article/details/79766690
        #    class_conf  [num_anchors, 1]  种类置信度
        #    class_pred  [num_anchors, 1]  种类
        #---------------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdims=True)

        # 利用置信度进行第一轮筛选
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        # 根据置信度进行预测结果的筛选
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #---------------------------------------------------------------#
        #    此时detections  [num_anchors, 7]
        #    7 = [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
        #---------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获得预测结果包含的所有种类
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # 获取某一类得分筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            #---------------------------------------------------------------#
            #    实现nms操作
            #---------------------------------------------------------------#
            # 按照存在物体的置信度排序
            _, conf_sort_index = torch.sort(detections_class[:, 4] * detections_class[:, 5], descending=True)
            detections_class = detections_class[conf_sort_index]
            # 进行非极大值抑制
            max_detections = []
            while detections_class.size(0):
                # 去除这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres, 如果是则去掉
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = box_iou(max_detections[-1], detections_class[1:], x1y1x2y2=True)
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠
            max_detections = torch.cat(max_detections).data

            # #---------------------------------------------------------------#
            # #    使用torchvison自带的nms进行筛选框
            # #---------------------------------------------------------------#
            # keep = nms(
            #     detections_class[:, :4],
            #     detections_class[:, 4] * detections_class[:, 5],
            #     nms_thres
            # )
            # max_detections = detections_class[keep]

            # 将经过nms后的检测框加入输出
            out_put[image_i] = max_detections if out_put[image_i] is None else torch.cat(
                (out_put[image_i], max_detections))
    return out_put


