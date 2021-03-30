# coding=utf-8
import os.path as osp
from easydict import EasyDict as edict
# project
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_PATH    = osp.join(PROJECT_PATH, 'data')

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# MODEL options
__C.MODEL = edict()

__C.MODEL.MODEL_TYPE = {"TYPE": "YOLOv4"}  # YOLO type: YOLOv4
__C.MODEL.ATTENTION  = {"TYPE": "NONE"}  # attention type: SEnet、CBAM or NONE

# YOLO options
__C.YOLO = edict()

__C.YOLO.MODEL_PATH       = 'model_data/last.pt'
__C.YOLO.CLASSES_PATH     = 'model_data/coco_classes.txt'
__C.YOLO.ANCHORS_PATH     = 'model_data/yolo_anchors.txt'
__C.YOLO.MODEL_IMAGE_SIZE = (416, 416, 3)
__C.YOLO.CONFIDENCE       = 0.3
__C.YOLO.NMS_IOU          = 0.3
__C.YOLO.CUDA             = True
__C.YOLO.LETTERBOX_IMAGE  = False


# detect

# train

# test
