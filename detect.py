'''
@Description:检测
@Author:Zigar
@Date:2021/03/11 11:44:19
'''
from PIL import Image
import os
from yolo import YOLO

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)

        r_image.show()