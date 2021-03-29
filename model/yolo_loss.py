'''
TODO 未debug
@Description:yolov4_loss 损失计算
@Author:Zigar
@Date:2021/03/11 11:32:42
'''
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils.utils import box_iou, box_ciou, clip_by_tensor, smooth_labels


#---------------------------------------------------------------#
#    均方误差损失函数(没有用到)
#    https://blog.csdn.net/oBrightLamp/article/details/85137756
#---------------------------------------------------------------#
def MSELoss(pred, target):
    return (pred - target) ** 2

#---------------------------------------------------------------#
#    二类交叉熵损失函数，torch.nn.BCELoss()
#    https://blog.csdn.net/geter_CS/article/details/84747670
#---------------------------------------------------------------#
def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


#---------------------------------------------------------------#
#    YOLOLoss 损失计算
#---------------------------------------------------------------#
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True, normalize=True):
        super(YOLOLoss, self).__init__()
        self.anchors        = anchors
        self.num_anchors    = len(anchors)
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes  # 5 = xywh + conf
        self.img_size       = img_size
        self.feature_length = [img_size[0]//32, img_size[0]//16, img_size[0]//8]
        self.label_smooth   = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf      = 1.0
        self.lamnda_cls       = 1.0
        self.lambda_loc       = 1.0
        self.cuda             = cuda
        self.normallize       = normalize

    def forward(self, input, targets=None):
        #--------------------------------------------------------------#
        #   input的shape为  batchsize, 3*(5+num_classes), 13, 13
        #                   batchsize, 3*(5+num_classes), 26, 26
        #                   batchsize, 3*(5+num_classes), 52, 52
        #--------------------------------------------------------------#
        # 每个batch的图片数量
        batch_size = input.size(0)
        # 特征层的宽高
        in_w, in_h = input.size(2), input.size(3)

#----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #---------------------------------------------------------------#
        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[0] / in_w

        #---------------------------------------------------------------#
        #    原始先验框大小是针对原始图片大小
        #    此时获得的scaled_anchors大小是相对于特征层的
        #---------------------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        #---------------------------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #   view():将一个多行的Tensor,拼接成一行
        #   https://blog.csdn.net/program_developer/article/details/82112372
        #   permute():将tensor的维度换位
        #   https://blog.csdn.net/york1996/article/details/81876886
        #   contiguous():把tensor变成在内存中连续分布的形式
        #   https://blog.csdn.net/qq_36653505/article/details/83375160
        #---------------------------------------------------------------#
        prediction = input.view(batch_size, int(self.num_anchors / 3),
                                self.bbox_attrs,in_h,in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 获取位置置信度，是否有目标
        pred_conf = torch.sigmoid(prediction[..., 4])

        # 获取种类置信度，是否为该种类
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-----------------------------------------------------------------#
        #   找到哪些先验框内部包含目标
        #   利用真实框和先验框计算交并比
        #   obj_mask    batch_size, 3, in_h, in_w   有目标的特征点
        #   no_obj_mask batch_size, 3, in_h, in_w   无目标的特征点
        #   t_box       batch_size, 3, in_h, in_w, 4   中心宽高的真实值
        #   t_conf      batch_size, 3, in_h, in_w   置信度真实值
        #   t_cls       batch_size, 3, in_h, in_w, num_classes  种类真实值
        #-----------------------------------------------------------------#
        obj_mask, no_obj_mask, t_box, t_conf, t_cls, box_loss_scale_x, box_loss_scale_y = self.get_target(
            targets, scaled_anchors, in_w, in_h
        )

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #---------------------------------------------------------------#
        no_obj_mask, pred_boxes_for_ciou = self.get_ignore(
            prediction, targets, scaled_anchors, in_w, in_h, no_obj_mask
            )
        
        #---------------------------------------------------------------#
        #    计算全部loss
        #---------------------------------------------------------------#
        if self.cuda:
            obj_mask, no_obj_mask = obj_mask.cuda(), no_obj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y = box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            t_conf, t_cls = t_conf.cuda(), t_cls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y

        # 计算预测结果与真实结果的CIOU的loss
        ciou = (1 - box_ciou(pred_boxes_for_ciou[obj_mask.bool()], t_box[obj_mask.bool()])) * box_loss_scale[obj_mask.bool()]
        loss_loc = torch.sum(ciou)

        # 计算置信度的loss
        loss_conf = torch.sum(BCELoss(pred_conf, obj_mask) * obj_mask) + \
                    torch.sum(BCELoss(pred_conf, obj_mask) * no_obj_mask)

        loss_cls = torch.sum(BCELoss(pred_cls[obj_mask == 1], smooth_labels(t_cls[obj_mask == 1], self.label_smooth, self.num_classes)))

        loss = loss_conf * self.lambda_conf + loss_cls * self.lamnda_cls + loss_loc * self.lambda_loc

        if self.normallize:
            num_pos = torch.sum(obj_mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = batch_size / 3

        return loss, num_pos

        


    #---------------------------------------------------------------#
    #    获取网络应该有的预测结果
    #    找到哪些先验框内部包含目标
    #    利用真实框和先验框计算交并比
    #    即找到真实框对应的先验框和网格点位置
    #---------------------------------------------------------------#
    def get_target(self, target, anchors, in_w, in_h):
        """
        @param:
        -------
        target:目标真实框
        anchors：先验框
        in_w, in_h:特征层的宽高
        @Returns:
        obj_mask:有目标的特征点
        no_obj_mask:无目标的特征点
        t_box:框位置xywh的真实值
        t_conf:置信度的真实值
        t_cls:类别的真实值
        box_loss_scale_x，box_loss_scale_y:回归loss的比例，让小目标的loss更大，大目标的loss更小
        -------

        """
        # 疑问：始终不知道输入的target是什么？
        # 答案：是目标真实框
        # 每个batch的图片数量
        batch_size = len(target)
        # 获得当前特征层先验框所属的编号，方便后面对先验框进行筛选
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        # 创建全是0，或全是1的矩阵
        obj_mask = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        no_obj_mask = torch.ones(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        t_x = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_y = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_w = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_h = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_box = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, 4, requires_grad=False)
        t_conf = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        t_cls = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        for b in range(batch_size):
            if len(target[b]) == 0:
                continue
            
            # target的值是小数的形式
            # 计算出正样本在特征层的中心点和宽高
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h
            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h

            # 计算出正样本属于特征层的哪个特征点
            # torch.floor()向下取整，对应的就是网格点的左上角位置
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)

            # 将真实框转换一个形式 num_true_box, 4
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs) ,gws, ghs], 1))

            #将先验框转换一个形式 9, 4
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))

            # 计算真实框与先验框交并比 num_true_box, 9
            anch_iou = box_iou(gt_box, anchor_shapes)

            #---------------------------------------------------------------#
            #    计算重合度最大的先验框是哪个
            #    torch.argmax():返回指定维度最大值的序号
            #    https://blog.csdn.net/weixin_42494287/article/details/92797061
            #---------------------------------------------------------------#
            best_anchs = torch.argmax(anch_iou, dim=-1)
            for i, best_anch in enumerate(best_anchs):
                if best_anch not in anchor_index:
                    continue
                #-------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框在特征层的x轴和y轴坐标
                #   gw和gh代表真实框在特征层的宽和高
                #-------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx, gy, gw, gh = gxs[i], gys[i], gws[i], ghs[i]

                if (gj<in_h) and (gi<in_w):
                    best_anch = best_anch - subtract_index
                    # no_obj_mask代表无目标的特征点
                    no_obj_mask[b, best_anch, gj, gi] = 0
                    # obj_mask代表有目标的特征点
                    obj_mask[b, best_anch, gj, gi] = 1
                    # t_x,t_y代表中心的真实值
                    t_x[b, best_anch, gj, gi] = gx
                    t_y[b, best_anch, gj, gi] = gy
                    # t_w,t_h代表中心的真实值
                    t_w[b, best_anch, gj, gi] = gw
                    t_h[b, best_anch, gj, gi] = gh

                    # 用于获得xywh的比例
                    # 大目标loss权重小，小目标loss权重大
                    box_loss_scale_x[b, best_anch, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_anch, gj, gi] = target[b][i, 3]

                    # t_conf,t_cls代表目标置信度和种类置信度
                    t_conf[b, best_anch, gj, gi] = 1
                    t_cls[b, best_anch, gj, gi, target[b][i, 4].long()] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        t_box[..., 0] = t_x
        t_box[..., 1] = t_y
        t_box[..., 2] = t_w
        t_box[..., 3] = t_h

        return obj_mask, no_obj_mask, t_box, t_conf, t_cls, box_loss_scale_x, box_loss_scale_y


    #---------------------------------------------------------------#
    #   将预测结果进行解码，判断预测结果和真实值的重合程度
    #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
    #   作为负样本
    #---------------------------------------------------------------#
    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, no_obj_mask):
        """
        @param:
        -------
        prediction:预测结果
        targets:目标真实框
        scaled_anchors:相对于特征层的先验框
        in_w, in_h:特征层的宽高
        no_obj_mask:无目标的特征点
        @Returns:
        -------
        no_obj_mask:需要忽略的负样本
        pred_boxes_for_ciou:网络的预测框
        """
        # 每个batch的图片数量
        batch_size = len(target)
        # 获得当前特征层先验框所属的编号，方便后面对先验框进行筛选
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数
        w = prediction[..., 2] # width
        h = prediction[..., 3] # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #---------------------------------------------------------------#
        #    生成网格，先验框中心，网格左上角
        #    torch.linspace():返回一个一维的tensor，
        #    这个张量包含了从start到end，分成steps个线段得到的向量
        #    https://blog.csdn.net/york1996/article/details/81671128
        #    repeat():沿着指定的维度重复tensor
        #---------------------------------------------------------------#
        
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(batch_size * self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(batch_size * self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(batch_size):
            pred_boxes_for_ignore = pred_boxes[i]
            # 将预测结果转换一个形式
            # pred_boxes_for_ignore      num_anchors, 4
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            # 计算真实框，并把真实框转换成相对于特征层的大小
            # gt_box      num_true_box, 4
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)

                # 计算真实框与预测框交并比 num_true_box, num_anchors
                anch_iou = box_iou(gt_box, pred_boxes_for_ignore)
                # 每个先验框对应真实框的最大重合度 anch_ious_max, num_anchors
                anch_iou_max, _ = torch.max(anch_iou, dim=0)
                anch_iou_max = anch_iou_max.view(pred_boxes[i].size()[:3])
                no_obj_mask[i][anch_iou_max > self.ignore_threshold] = 0
        return no_obj_mask, pred_boxes
