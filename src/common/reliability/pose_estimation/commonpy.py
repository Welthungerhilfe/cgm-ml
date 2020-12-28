import pandas as pd
import os,sys
import random
import numpy as np
import cv2 as cv
import re

from azureml.core import Workspace, Datastore, Dataset

def getMountContext(path):
  ws = Workspace.from_config()
  ws
  dataset = Dataset.get_by_name(ws, name=path)
  type(dataset)

  mount_context = dataset.mount()
  mount_context.start()  # this will mount the file streams
  print(mount_context.mount_point)
  return mount_context

#dismount/stop the mountcontext
def stopMountContext(mountcontext):
  mountcontext.stop()

def getTitleboxAndResizeFrame(frame, scanArtifactName, additionalTitleText = ''):
  #create box - white no color
  brdColor = [255,255,255]     #---Color of the border---
  #copyMakeBorder -- top, bottom, left, right
  frameWBorder=cv.copyMakeBorder(frame,10,10,30,1290,cv.BORDER_CONSTANT,value=brdColor)
  #print('frameWBorder size ', frameWBorder.shape[1])
  #add title text - for description
  titlebox = np.zeros((350, frameWBorder.shape[1], 3), np.uint8)
  titlebox[:] = (255, 255, 255) 
  
  #print('scanArtifactName ', scanArtifactName)
  resTitle = 'Pose estimation for scan\n' + scanArtifactName
  if (additionalTitleText != ''):
    resTitle = resTitle + '\n' + additionalTitleText
    
  position = (10, 45)
  font_scale = 1.8
  color = (0, 0, 0)
  thickness = 3
  font = cv.FONT_HERSHEY_SIMPLEX
  line_type = cv.LINE_AA

  text_size, _ = cv.getTextSize(resTitle, font, font_scale, thickness)
  line_height = text_size[1] + 5
  x, y0 = position
  for i, line in enumerate(resTitle.split("\n")):
    y = y0 + 32 + (i * line_height)
    cv.putText(titlebox, line, (x, y), font, font_scale, color,
                  thickness, line_type)

  return frameWBorder, titlebox

def getScansFiles(dsName, pose, numfiles):
  # mount_context
  mount_context = getMountContext(dsName)
  print('mount_context= ',mount_context) 
  # directory
  direc_scans = mount_context.mount_point + '/scans'
  # list directory
  folders_scans = os.listdir(direc_scans)
  random.shuffle(folders_scans)
  
  folderlen = len(folders_scans)
  folderToUse = 0
  if (folderlen > 1):
    # randomize the choice of folder
    random_files = np.random.choice(folders_scans, 2)
    folderToUse = folderlen - 1
  
  folderatIdx = folders_scans[folderToUse]
  folders_scans = os.listdir(direc_scans)
  
  path = os.path.join(direc_scans, folderatIdx)+'/' + str(pose);
  print('scans path - ', path)
  scans = []
  random_files = []
  scanFilenamesOnly = []

  for root, dirs, files in os.walk(path):
    # randomize the choice of files
    random_files = np.random.choice(files, numfiles) 
    
    for name in random_files:
      scanFilenamesOnly.append(name)
      scans.append(os.path.join(root, name))

  #return scan files array, mount_context and scanfilenames only
  return scans, mount_context, scanFilenamesOnly

def getRandomScans(dsName, numfiles):
  # mount_context
  mount_context = getMountContext(dsName)
  print(mount_context)

  #scans folder
  direc_scans = mount_context.mount_point + '/scans'
  # list directory
  folders_scans = os.listdir(direc_scans)
  random.shuffle(folders_scans)
  
  scanFilenamesOnly = []
  scans = []
  
  # randomize the choice of folders - so we get at least numfiles number
  # of folders
  random_qrs = np.random.choice(folders_scans, numfiles)
  
  randomSize = 1
  for qrcodePath in random_qrs:
    dirpath = os.path.join(direc_scans, qrcodePath)
    paths = os.listdir(dirpath)
    #print(paths)
    path = np.random.choice(paths, 1)[0]
    
    path = os.path.join(dirpath, path)
    #print('path aft', path)

    for root, dirs, files in os.walk(path):
        # randomize the choice of files
        random_files = np.random.choice(files, randomSize) 

        for name in random_files:
            scanFilenamesOnly.append(name)
            scans.append(os.path.join(root, name))

  return scans, mount_context, scanFilenamesOnly
