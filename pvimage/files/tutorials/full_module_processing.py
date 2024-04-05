#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
@author: jlbraid
"""
import pvimage as pvi
import glob2
from os import chdir
import os
import cv2
from matplotlib import pyplot as plt
wd=os.path.dirname(__file__)
chdir(wd)

folder_path = '../data/FullSizeModules/*'
files = glob2.glob(folder_path)
save_path = '../data/out/'

#Applying lens correction, planar indexing the module, and extracting cells
for file in files:
    n,f = pvi.pipelines.GetLensCorrectParams(file)
    #If a large batch of images, get average LC parameters for a subset of good
    #images, and apply to images with the same camera setup
    pvi.pipelines.FMpipeline(file,save_path,n,f,10,6,True)

#Demonstrating the pipeline
img = cv2.imread(files[0])
plt.imshow(img)
plt.show()

n,f = pvi.pipelines.GetLensCorrectParams(files[0])

lc = pvi.process.lensCorrect(img, n, f)
plt.imshow(lc)
plt.show()

mask = pvi.process.Mask(lc)
plt.imshow(mask)
plt.show()

pi = pvi.process.PlanarIndex(lc)
plt.imshow(pi)
plt.show()

cells = pvi.process.CellExtract(pi, 10, 6)

for cell in cells:
    plt.imshow(cell)
    plt.show()
