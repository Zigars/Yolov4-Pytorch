'''
@Description:创建YOLO类
@Author:Zigar
@Date:2021/03/11 11:40:42
'''
import colorsys
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from model.yolov4 import Yolo_Body
from utils.utils import DecodeBox,  letterbox_image, yolo_correct_boxes, non_max_suppression
from config.config import cfg


#---------------------------------------------------------------#
#    创建YOLO类
#---------------------------------------------------------------#
class YOLO(object):
    #---------------------------------------------------------------#
    #    初始化YOLO
    #---------------------------------------------------------------#
    def __init__(self, **kwargs):
        # 初始化参数
        self.model_path       = cfg.YOLO.MODEL_PATH
        self.classes_path     = cfg.YOLO.CLASSES_PATH
        self.anchors_path     = cfg.YOLO.ANCHORS_PATH
        self.model_image_size = cfg.YOLO.MODEL_IMAGE_SIZE
        self.confidence       = cfg.YOLO.CONFIDENCE
        self.nms_iou          = cfg.YOLO.NMS_IOU
        self.cuda             = cfg.YOLO.CUDA
        self.letterbox_image  = cfg.YOLO.LETTERBOX_IMAGE

        self.class_names = self._get_class()
        self.anchors     = self._get_anchors()
        self.generate()

    #---------------------------------------------------------------#
    #    获得所以的分类
    #---------------------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        # print(class_names)
        return class_names

    #---------------------------------------------------------------#
    #    获取所有的先验框
    #---------------------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser( self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]
        # print(len(anchors[0]))
        return anchors


    #---------------------------------------------------------------#
    #    生成模型
    #---------------------------------------------------------------#
    def generate(self):
        # 建立yolov4模型
        self.model = Yolo_Body(len(self.anchors[0]), len(self.class_names)).eval()

        # 载入yolov4模型的权重
        print('Loading weight into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        # 建立三个特征层解码用的工具
        self.yolo_decodes = []

        for i in range(3):
            self.yolo_decodes.append(DecodeBox(
                self.anchors[i],
                len(self.class_names),
                (self.model_image_size[1], self.model_image_size[0])
                ))
        print('{} model, anchors, and classes load.'.format(self.model_path))


    #---------------------------------------------------------------#
    #    检测图片
    #---------------------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # w, h = self.model_image_size[0], self.model_image_size[1]
        #---------------------------------------------------------------#
        #    给图像增加灰边，实现不失真的resize
        #    也可以直接resize进行识别
        #---------------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255
        photo = np.transpose(photo, (2, 0, 1))

        #---------------------------------------------------------------#
        #    添加batch_size维度
        #---------------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if  self.cuda:
                images = images.cuda()

            #---------------------------------------------------------------#
            #    将图片输入网络当中进行预测
            #---------------------------------------------------------------#
            out_puts = self.model(images)
            out_put_list = []
            for i in range(3):
                out_put_list.append(self.yolo_decodes[i](out_puts[i]))

            #---------------------------------------------------------------#
            #    将预测框进行堆叠后，进行非极大值抑制nms
            #---------------------------------------------------------------#
            out_put = torch.cat(out_put_list, 1)
            batch_detections = non_max_suppression(
                out_put, len(self.class_names),
                conf_thres = self.confidence,
                nms_thres = self.nms_iou
            )

            #---------------------------------------------------------------#
            #    如果没有检测到目标，就返回原图
            #---------------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            #---------------------------------------------------------------#
            #    对预测框进行置信度筛选
            #    在nms里已经进行了一次置信度筛选，这里好像不需要？
            #    验证后，这里conf阈值为0结果也一样，所以修改了这部分代码
            #---------------------------------------------------------------#
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            # print(top_index)
            top_conf   = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label  = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin   = np.expand_dims(top_bboxes[:, 0], -1)
            top_ymin   = np.expand_dims(top_bboxes[:, 1], -1)
            top_xmax   = np.expand_dims(top_bboxes[:, 2], -1)
            top_ymax   = np.expand_dims(top_bboxes[:, 3], -1)

            #---------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分
            #---------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(
                    top_ymax, top_xmin, top_ymax, top_xmax,
                    np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape
                    )
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                # top, left, bottom, right = top_ymin, top_xmin, top_ymax, top_xmax
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
                # print(boxes)

            #---------------------------------------------------------------#
            #    绘制检测框
            #---------------------------------------------------------------#
            # 检测框不同颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

            # 字体
            font = ImageFont.truetype("arial.ttf", 24)

            # # 检测框的厚度
            # # 选择去掉是因为不加的话在小图片显示更清晰
            # thick_ness = max((np.shape(image)[0] + np.shape(image)[1]) // w, 1)

            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]
                score = top_conf[i]

                top, left, bottom, right = boxes[i]
                # 略微扩大检测框，使得目标被包裹
                # top, left, bottom, right = top - 5, left - 5, bottom - 5, right - 5

                # 保证检测框在图片范围
                top    = max(0, np.floor(top + 0.5).astype('int32'))
                left   = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right  = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

                # 绘制检测框
                label = '{} {:.2f}'.format(predicted_class, score)
                print(label, top, left, bottom, right)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                # label对应检测框的位置
                # 如果top在图片顶部，就把label画在检测框里面
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # for i in range(thick_ness):
                # 绘制检测框
                draw.rectangle(
                    [left, top, right, bottom],
                    outline = colors[self.class_names.index(predicted_class)]
                    )

                # 绘制label框
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill = colors[self.class_names.index(predicted_class)]
                )

                # 绘制label
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            return image


# if __name__ == '__init__':
#     YOLO()
