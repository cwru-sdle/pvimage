#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
@author: Jennifer Braid
"""
import pvimage as pvi
import glob2
from os import chdir
import os
import cv2
from matplotlib import pyplot as plt
wd=os.path.dirname(__file__)
chdir(wd)

ELfolder_path = '../data/Minimodules/*EL*'
ELfiles = glob2.glob(ELfolder_path)
save_path = '../data/out/'

#Extracting individual cells, saving each, and stitching into a combined image
for file in ELfiles:
    try:
        pvi.pipelines.MMpipeline(file,save_path,2,2,True)
    except OverflowError:
        pvi.pipelines.MMpipeline(file,save_path,2,2,True,'lowcon')

PLfile = '../data/Minimodules/MMPL.tiff'
pvi.pipelines.MMpipeline(PLfile,save_path,2,2,True,'lowcon')

saved = glob2.glob('../data/out/*')
for im in saved:
    plt.imshow(cv2.imread(im))
    plt.show()

#Demonstrating the pipeline
img = cv2.imread(PLfile)
plt.imshow(img)
plt.show()

mask = pvi.process.Mask(img, 'lowcon')
plt.imshow(mask)
plt.show()

cells = pvi.process.CellExtract(img, 2, 2)

for cell in cells:
    plt.imshow(cell)
    plt.show()
    plt.imshow(pvi.process.Mask(cell,'lowcon'))
    plt.show()

planarindexed = []
for cell in cells:
    planarindexed.append(pvi.process.PlanarIndex(cell,'lowcon'))

for cell in planarindexed:
    plt.imshow(cell)
    plt.show()

