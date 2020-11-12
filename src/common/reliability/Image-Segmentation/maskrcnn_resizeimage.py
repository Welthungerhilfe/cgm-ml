#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0, "utils")
from utils.ipstartup import *
import numpy as np
from imgseg.predict import predict, show
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import time
from IPython.display import display

# In[427]:


#get_ipython().system('ls trainrgb/scans/1583462481-e4vbd8pnrg/100/rgb_1583462481-e4vbd8pnrg_1591122197970_100_1029804.6101950521.jpg')


# In[428]:


model = maskrcnn_resnet50_fpn(pretrained=True)


# In[429]:


#Applies MaskRCNN on downscaled image by 10x
def predictByResize(image, flag=0, factor=10):
    print("Original Image Dimension: ", image.size)
    rimage = image
    if flag == 1 :
        newsize = (int(image.size[0] / factor), int(image.size[1] / factor))
        rimage = image.resize(newsize)
    start_time = time.time()
    print("After Resizing, Image Dimension: ", rimage.size)
    out = predict(rimage, model)
    print("Time: %s s" % (time.time() - start_time))
    return (rimage, out)


# In[430]:


#File path
f = r"trainrgb/scans/1583462481-e4vbd8pnrg/100/rgb_1583462481-e4vbd8pnrg_1591122197970_100_1029804.6101950521.jpg"

#Load Image
image = Image.open(f)

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


# In[431]:


#Original Image
display(outputImage[0])


# In[432]:


#Display the mask region, by default
width, height = outputImage[0].size
blank_img_arr = np.zeros((height, width, 3), np.uint8)
blank_img = Image.fromarray(blank_img_arr, 'RGB')
show(blank_img, outputImage[1], alpha=0.9999)  # Returns an binary image


# In[434]:


#File path
f = r"trainrgb/scans/1585003291-npaa8l8yxt/200/rgb_1585003291-npaa8l8yxt_1597112683680_200_821.531431559.jpg"

#Load Image
image = Image.open(f)

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


# In[435]:


#Original Image
display(outputImage[0])


# In[436]:


#Display the mask region, by default
width, height = outputImage[0].size
blank_img_arr = np.zeros((height, width, 3), np.uint8)
blank_img = Image.fromarray(blank_img_arr, 'RGB')
show(blank_img, outputImage[1], alpha=0.9999)  # Returns an binary Image


# In[437]:


#f = r"trainrgb/scans/1585003291-npaa8l8yxt/200/rgb_1585003291-npaa8l8yxt_1597112683680_200_821.531431559.jpg"
#f = r"trainrgb/scans/1585003414-1qlf0qlv29/100/rgb_1585003414-1qlf0qlv29_1591708847031_100_876217.1931517591.jpg"
f = r"trainrgb/scans/1583438084-zkafuhr4xx/100/rgb_1583438084-zkafuhr4xx_1591122031563_100_166438.52068312402.jpg"
image = Image.open(f)
image = image.rotate(270, expand=1)
outputImage = predictByResize(image, 1, 5)


# In[438]:


#Original Image
display(outputImage[0])


# In[439]:


#Display the mask region, by default
width, height = outputImage[0].size
blank_img_arr = np.zeros((height, width, 3), np.uint8)
blank_img = Image.fromarray(blank_img_arr, 'RGB')
show(blank_img, outputImage[1], alpha=0.9999)  # Returns an binary image