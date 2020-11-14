#!/usr/bin/env python
# coding: utf-8
import sys
# from utils.ipstartup import *
import numpy as np
from imgseg.predict import predict, show
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import time
from IPython.display import display
sys.path.insert(0, "utils")

model = maskrcnn_resnet50_fpn(pretrained=True)

#Applies MaskRCNN on downscaled image, by default the factor is 10x
def predictByResize(image, flag=0, factor=10):
    print("Original Image Dimension: ", image.size)
    rimage = image
    if flag == 1:
        newsize = (int(image.size[0] / factor), int(image.size[1] / factor))
        rimage = image.resize(newsize)
    start_time = time.time()
    print("After Resizing, Image Dimension: ", rimage.size)
    out = predict(rimage, model)
    print("Time: %s s" % (time.time() - start_time))
    return (rimage, out)

#File path
def processImage(filepath):
    f = filepath

    #Load Image
    image = Image.open(f)
    #image = image.rotate(270, expand = 1)  # For rotation

    #Prediction
    outputImage = predictByResize(image, flag=1, factor=5)  # Applying MaskRCNN

    #Getting the masked region
    region = outputImage[1]
    mask_area = int(np.reshape(region['masks'], (-1, region['masks'].shape[-1])).astype(np.float32).sum())

    #Mask Stats like percentage of body covered, mask area
    perc_body_covered = (mask_area * 100) / (outputImage[0].size[0] * outputImage[0].size[1])
    perc_body_covered = round(perc_body_covered, 2)
    print("Mask Area:", mask_area, "px")
    print("Percentage of body covered to total pixels:", perc_body_covered, "%")



#To run the function
#processImage(r"trainrgb/scans/1583462481-e4vbd8pnrg/100/rgb_1583462481-e4vbd8pnrg_1591122197970_100_1029804.6101950521.jpg")
