#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:11:32 2020

@author: Ben, NormanJost
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage import measure
from scipy import stats
import glob
from skimage import io
from skimage import transform
import scipy
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from sklearn.cluster import AgglomerativeClustering
import os
import pandas as pd
import re


def feature_extraction_crack_mask(images, dfinfo):  
    """
    This function extracts features from a list of images and appends the features to a DataFrame.

    Args:
        images: List of images to process.
        dfinfo: DataFrame containing additional image information.

    Returns:
        DataFrame containing extracted features.
    """
    dffeatures = pd.DataFrame()
    testimages = images
    
    for n, testimage in enumerate(testimages): #testimages
        gray = testimage
        gimg = cv2.GaussianBlur(gray,(3,3),0) #11,11 #small works better with bigger images
        gimg = gimg**(1/2.)
        gimg = cv2.normalize(gimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        gimg = gimg.astype(np.uint8)
        th2 = cv2.adaptiveThreshold(gimg, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,2) #original: 51,2
        # plt.imshow(th2, cmap='gray')
        
        kernel = np.ones((20,20), np.uint8)
        blackhat = cv2.morphologyEx(gimg, cv2.MORPH_BLACKHAT, kernel)
        # plt.figure(figsize=(3,3))
        # plt.imshow(blackhat, cmap='gray')
        kernel = np.ones((3,3), np.uint8)
        blackhat = cv2.morphologyEx(blackhat,cv2.MORPH_OPEN, kernel)
        mask = cv2.threshold(blackhat, blackhat.max()/8, blackhat.max(), cv2.THRESH_BINARY)[1]
        mask2 = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
        mask2n = mask2/mask2.max()
        # plt.figure(figsize=(3,3))
        # plt.imshow(mask2n, cmap='gray')
        
        sk = skeletonize(mask2n)
        # plt.figure(figsize=(3,3))
        # plt.imshow(sk, cmap='gray')
        
        sk[0:5,:]=False
        sk[-5:-1,:]=False
        sk[:,0:5]=False
        sk[:,-5:-1]=False
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        sk_enhanced  = cv2.dilate(sk.astype('uint8'), kernel, iterations=3)
        vector = sk.sum(axis = 0)
        signal = (vector > np.roll(vector,1)) & (vector > np.roll(vector,-1))
        top4 = (-vector[signal]).argsort()[:4] # get argmax largest values
        top4_idx = signal.nonzero()[0][top4] # get their indexes
        
        blackhat = sk_enhanced
        # plt.figure(figsize=(3,3))
        # plt.imshow(blackhat)
        
        bbrmd2 = blackhat
        bbrmd2 = cv2.dilate(bbrmd2.astype('uint8'), kernel, iterations=2)
        # plt.figure(figsize=(3,3))
        # plt.imshow(bbrmd2)
        
        opened = opening(bbrmd2)

        props = extract_img_feats(bbrmd=bbrmd2, imgsrc=dfinfo.impath[n], top_n = 'all', by = 'perimeter', show = False, min_thresh = 100)
        propnames = 'cell_number, i, prop.perimeter, slope, prop.convex_area, prop.area, prop.orientation, prop.image'

        listprops = propnames.split(',')

        for i in range(0, len(props)):
            names = []
            values = []
            for s1,s2 in zip(listprops, props[i]):
                names.append(s1)
                values.append(s2)
            dftemp = pd.DataFrame(data = values[:-1]).T
            dftemp.columns = names[:-1]
            dffeatures = pd.concat([dffeatures, dftemp], axis=0)
    return dffeatures

def extract_cracks(gray, imgsrc, busbar_orient = 'vertical', iteration = 3, pix_thresh = 4, show = False):
    """
    This function extracts cracks from a grayscale image.

    Args:
        gray: Grayscale image to process.
        imgsrc: Source of the image.
        busbar_orient: Orientation of busbars ('vertical' or 'horizontal').
        iteration: Number of iterations for dilation.
        pix_thresh: Pixel threshold for removing busbars.
        show: Whether to display intermediate images.

    Returns:
        Processed image with cracks extracted.
    """
    if busbar_orient == 'horizontal':
        gray = transform.rotate(gray,90)
    
    
    gimg = cv2.GaussianBlur(gray,(11,11),0)
    gimg = gimg**(1/2.)
    gimg = cv2.normalize(gimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    th2 = cv2.adaptiveThreshold(gimg, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,2)
    kernel = np.ones((15,15), np.uint8)
    blackhat = cv2.morphologyEx(gimg, cv2.MORPH_BLACKHAT, kernel)
    kernel = np.ones((7,7), np.uint8)
    blackhat = cv2.morphologyEx(blackhat,cv2.MORPH_OPEN, kernel)
    mask = cv2.threshold(blackhat, blackhat.max()/8, blackhat.max(), cv2.THRESH_BINARY)[1]
    mask2 = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    #blackhat = cv2.morphologyEx(blackhat,cv2.MORPH_OPEN, kernel)
    #blackhat2= (blackhat > blackhat.max()/10)*blackhat
    mask2n = mask2/mask2.max()
    
    sk = skeletonize(mask2n)
    
    sk[0:5,:]=False
    sk[-5:-1,:]=False
    sk[:,0:5]=False
    sk[:,-5:-1]=False
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    sk_enhanced  = cv2.dilate(sk.astype('uint8'), kernel, iterations=5)
    vector = sk.sum(axis = 0)
    signal = (vector > np.roll(vector,1)) & (vector > np.roll(vector,-1))
    top4 = (-vector[signal]).argsort()[:4] # get argmax largest values
    top4_idx = signal.nonzero()[0][top4] # get their indexes
    
#    pix_thresh = 5
#    bbar_rm = np.copy(sk)
#    for local_max in top4_idx:
#        for i in range(local_max-pix_thresh,local_max+pix_thresh):
#            bbar_rm[:,i] = 0
#            
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#    bbrmd  = cv2.dilate(bbar_rm.astype('uint8'), kernel, iterations=iteration)
    blackhat = mask2n
    # Removing Busbars:
    col_mean = np.zeros(blackhat.shape[1])
    for i in range(blackhat.shape[1]):
        col_mean[i] = np.mean(blackhat[:,i])
    nbb = 4
    peaks = scipy.signal.find_peaks(col_mean,distance=blackhat.shape[1]/(nbb+1))[0]
    [mintab,maxtab] = scipy.signal.peak_widths(col_mean,peaks,rel_height=.9)[2:4]
    for i in range(len(mintab)):
        left = int(mintab[i])
        right = int(maxtab[i])
        blackhat[:,range(left,right+1)]=0
    #plt.imshow(blackhat,cmap='gray')
    bbrmd2 = blackhat
    bbrmd2 = cv2.dilate(bbrmd2.astype('uint8'), kernel, iterations=iteration)
    #plt.imshow(bbrmd)
    
    opened = opening(bbrmd2)
    
    if show:
        plt.imshow(gray, cmap = 'Greys_r')
        plt.title(imgsrc)
        plt.show()
#        plt.imshow(gimg, cmap = 'Greys_r')
#        plt.title(imgsrc)
#        plt.show()
#        plt.imshow(blackhat, cmap = 'Greys_r')
#        plt.title(imgsrc)
#        plt.show()
        plt.imshow(mask2n, cmap = 'Greys_r')
        plt.title(imgsrc)
        plt.show()
        plt.imshow(sk, cmap = 'Greys_r')
        plt.title(imgsrc)
        plt.show()
        plt.imshow(bbrmd2, cmap = 'Greys_r')
        plt.show()
        plt.imshow(opened, cmap = 'Greys_r')
        plt.show()
    return bbrmd2

def extract_img_feats(bbrmd, imgsrc, top_n = 'all', by = 'perimeter', show = False, min_thresh = 500):
    """
    This function extracts image features from a labeled image.

    Args:
        bbrmd: Binary image with labeled regions.
        imgsrc: Source of the image.
        top_n: Number of top features to return.
        by: Criterion to sort the features ('area' or 'perimeter').
        show: Whether to display the labeled image.
        min_thresh: Minimum threshold for feature extraction.

    Returns:
        List of extracted features.
    """
    all_labels = measure.label(bbrmd, background=0, connectivity = 2)
    properties = measure.regionprops(all_labels, cache = True)
    properties_key= []
    if show:
        plt.imshow(bbrmd, cmap = 'Greys_r')
        plt.title(imgsrc)
        plt.show()
    for i in range(len(properties)):
        if by == 'area':
            if properties[i].area > min_thresh:
                properties_key.append((properties[i],properties[i].area, i))
            else:
                continue
        elif by == 'perimeter':
            #print(properties[i].perimeter)
            if properties[i].perimeter > min_thresh:
                properties_key.append((properties[i],properties[i].perimeter, i))
            else:
                continue
    if top_n == 'all':
        properties_sorted = sorted(properties_key, key = lambda x: x[1])[::-1]
    else:
        properties_sorted = sorted(properties_key, key = lambda x: x[1])[::-1][:top_n]

    #print(properties_sorted)
    crackprops = []
    for crack in properties_sorted:
        prop, area, i = crack
        slope, intercept, r_value, p_value, std_err = stats.linregress(prop.coords)
        file_name,ext = os.path.splitext(os.path.split(imgsrc)[1])
        
        cell_number = re.split(r"-", file_name)[0]
        # print(cell_number)
        
        crackprops.append([cell_number, i, prop.perimeter, slope, prop.convex_area, prop.area, prop.orientation, prop.image])
    return crackprops

def extract_all_feats(imgpath, filetype = '.tiff', postfix = "EL_9", show = False):
    path = '{fp}*/*{postfix}*{filetype}'.format(fp=imgpath, postfix = postfix, filetype = filetype)
    #print(glob.glob(path,recursive = True))
    all_feats = []
    for imgpth in glob.glob(path,recursive = True):
        gray = io.imread(imgpth, as_gray = True)
        bbrmd = extract_cracks(gray,imgpth, show = show)
        feats = extract_img_feats(bbrmd,imgpth, show = show)
        for f in feats:
            all_feats.append(f)
    return all_feats  