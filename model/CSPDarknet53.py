'''
Description: backbone骨干网络
Author: Zigar
Date: 2021-03-04 16:45:31
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import Convolutional, Resblock


#---------------------------------------------------------------#
#    Resblock_body
#    CSPdarknet的结构块
#    1、利用步长为2x2的卷积块进行高和宽的压缩(下采样操作)
#    2、建立一个大的残差边，这个大残差边绕过所有的小残差块
#    3、主干部分会对num_block进行循环，循环内部是小残差块
#    4、CSPdarknet的结构块 = 外部大残差块 + 内部多个小残差块
#---------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        #---------------------------------------------------------------#
        #   下采样操作(降低图片分辨率)
        #   利用一个步长为2x2的卷积块进行高和宽的压缩
        #---------------------------------------------------------------#
        self.downsample_conv = Convolutional(in_channels, out_channels, 3, stride=2)

        #---------------------------------------------------------------#
        #    首层和其它层Resblock_body的不同点：
        #    1、首层结构的split-conv通道数“未减半”，在隐层“减半”
        #      其它层结构的split-conv通道数“减半”，隐层“未减半”
        #    2、首层结构的concat后通道数“减半”
        #       其他层结构的concat后通道数“未减半”
        #    首层和其它层Resblock_body的相同点：
        #    2、输出通道数 = 2 * 输入通道数相同    
        #---------------------------------------------------------------#
        if first:
            #---------------------------------------------------------------#
            #    CSP_1
            #    首个Resblock_body结构
            #    包含一个大的残差边和一个小残差块
            #---------------------------------------------------------------#
            self.split_conv0 = Convolutional(out_channels, out_channels, 1)

            self.split_conv1 = Convolutional(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(out_channels, out_channels//2),
                Convolutional(out_channels, out_channels, 1)
            )
            self.concat_conv = Convolutional(out_channels * 2, out_channels, 1)
        else:
            #---------------------------------------------------------------#
            #    CSP_n
            #    其它Resblock_body结构
            #    包含一个大的残差边和多个小残差块  
            #---------------------------------------------------------------#
            self.split_conv0 = Convolutional(out_channels, out_channels//2, 1)

            self.split_conv1 = Convolutional(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                Convolutional(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = Convolutional(out_channels, out_channels, 1)

    def forward(self, x):
        # 下采样
        x = self.downsample_conv(x)
        # 大残差边
        x0 = self.split_conv0(x)
        # 小残差块
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        # 大残差边和小残差块的输出张量concat（通道堆叠）
        x = torch.cat([x1, x0], dim=1)
        # 整合通道数
        x = self.concat_conv(x)

        return x


#---------------------------------------------------------------#
#    CSPdarknet53的主体部分
#    输入为一张416x416x3的图片
#    输出为三个有效特征层
#---------------------------------------------------------------#
class CSPdarknet(nn.Module):
    def __init__(self, layers):
        super(CSPdarknet, self).__init__()
        self.stem_channels    = 32
        self.feature_channels = [64, 128, 256, 512, 1024]

        # 416,416,3 -> 416,416,32
        self.stem_conv = Convolutional(3, self.stem_channels, kernel_size=3, stride=1)

        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            Resblock_body(self.stem_channels, self.feature_channels[0], layers[0], first=True),
            # 208,208,64 -> 104,104,128
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            # 104,104,128 -> 52,52,256
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            # 52,52,256 -> 26,26,512
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            # 26,26,512 -> 13,13,1024
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self._initialize_weights()

    def forward(self, x):
        # feature_channels = self.feature_channels
        x = self.stem_conv(x)

        x = self.stages[0](x)
        x = self.stages[1](x)

        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return [out3, out4, out5]
    
    #---------------------------------------------------------------#
    #    初始化权重
    #---------------------------------------------------------------#
    def _initialize_weights(self):
        print("**"*10, "initing CSPDarknet53 weights", "**"*10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                print("initing {}".format(m))


#---------------------------------------------------------------#
#    构建CSPdarknet模型
#---------------------------------------------------------------#
def CSPdarknet53(pretrained):
    model = CSPdarknet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got[{}]".format(pretrained))
    return model


#---------------------------------------------------------------#
#    test model
#---------------------------------------------------------------#
if __name__ == '__main__':
    model = CSPdarknet53(None)
    x = torch.randn(1, 3, 416, 416)
    y = model(x)
    print(x.shape)
    for i in range(3):
        print(y[i][0].shape)