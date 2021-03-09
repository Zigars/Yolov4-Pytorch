from model.head import yolo_head
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.neck import *
from model.head import *
from model.yolov4 import Yolo_Body

#---------------------------------------------------------------#
#    test model
#---------------------------------------------------------------#
# if __name__ == '__main__':
#     model = CSPdarknet53(None)
#     x = torch.randn(1, 3, 416, 416)
#     y = model(x)
#     print(x.shape)
#     for i in range(3):
#         print(y[i][0].shape)

if __name__ == "__main__":
    feature_channels = [64, 128, 256, 512, 1024]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(0) if cuda else "cpu")
    backbone = CSPdarknet53(None).to(device)
    # print(backbone)
    neck = yolo_neck(feature_channels).to(device)
    # print(neck)
    heads = yolo_head(255, feature_channels).to(device)
    # print(head)
    yolov4 = Yolo_Body(3, 80).to(device)
    torch.save(yolov4.state_dict(), 'yolov4.pt')
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
    print("*"*8 , "this is head output", "*"*8)
    features = heads(features)
    print(features[0][0].shape)
    print(features[1][0].shape)
    print(features[2][0].shape)
    print("*"*8 , "this is yolov4 output", "*"*8)
    features = yolov4(x)
    print(features[0][0].shape)
    print(features[1][0].shape)
    print(features[2][0].shape)
        
