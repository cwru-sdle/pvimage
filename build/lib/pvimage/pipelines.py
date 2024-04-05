#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:29:52 2020

@author: jlbraid
"""
import os
import cv2
from pvimage import process
import numpy as np
from skimage import io
import glob
from skimage import transform

# Finds lens distortion correction parameters for a given image
def GetLensCorrectParams(imagepath,imgtype=''):
    """Automatically detects lens correction parameters for an image
        using linear fitting of the module edges.

    Args:
        imagepath (str): path to a raw image
        imgtype (str):  'UVF' UV Fluorescence Image
                        'gradient' for unequal intensity across the image
                        'lowcon' low contrast between cell and background
                            or PL images without background subtraction
    Returns:
        n,f (float): lens correction parameters
    """    
    img = cv2.imread(imagepath)
    mask = process.Mask(img,imgtype)
    return process.autoLensCorrect(mask)

def MMpipeline(imagepath,savepath,numCols,numRows, stitch=False, imgtype='', changepoints = False):
    """Performs minimodule image processing steps, including cell extraction,
        planar indexing, and re-combination of cell images into a module
        image, if desired.

    Args:
        imagepath (str): path to a raw image
        savepath (str): folder path for saving output
        numCols (int): number of cells across in module image
        numRows (int): number of cells down in module image
        stitch (bool): True if output image with cell-level images stitched
            together is desired. Default is False
        imgtype (str):  'UVF' UV Fluorescence Image
                        'gradient' for unequal intensity across the image
                        'lowcon' low contrast between cell and background
                            or PL images without background subtraction
    Returns:
        
    """    
    #img = cv2.imread(imagepath)
    img = io.imread(imagepath)
    file_name,ext = os.path.splitext(os.path.split(imagepath)[1])
    rotated = process.RotateImage(img,imgtype)
    if changepoints:
        cellarrays, col_ch_pts, row_ch_pts = process.CellExtract(rotated,numCols,numRows,imgtype, changepoints = True)
    else:
        cellarrays = process.CellExtract(rotated,numCols,numRows,imgtype)
    pi = []
    if len(cellarrays)>1:
        for i in range(len(cellarrays)):
            planarindexed = process.PlanarIndex(cellarrays[i],imgtype)
            image_name = savepath+file_name+'-c'+ '{:02}'.format(i+1) + ext
            cv2.imwrite(image_name,planarindexed)
            pi.append(planarindexed)
        if stitch == True:
            dims = []
            for array in pi:
                y,x = array.shape[0:2]
                dims.append([y,x])
            dims = np.asarray(dims)
            newy = int(np.mean(dims[:,0]))
            newx = int(np.mean(dims[:,1]))
            resized = []
            for array in pi:
                resized.append(cv2.resize(array,(newy,newx)))
            top = np.concatenate((resized[0],resized[1]),axis=1)
            bot = np.concatenate((resized[2],resized[3]),axis=1)
        
            tot = np.concatenate((top,bot),axis=0)
            save_name = savepath+file_name + ext
            cv2.imwrite(save_name,tot)
    elif len(cellarrays)==1:
        out = process.PlanarIndex(cellarrays[0],imgtype)
        image_name = savepath+file_name + ext
        cv2.imwrite(image_name,out)
        
    if changepoints:
        return col_ch_pts, row_ch_pts
    return True

def FMpipeline(imagepath,savepath,n=None,f=None,numCols=None,numRows=None,savesmall=False,imgtype=''):
    """Performs full-size module image processing steps, including
        lens correction, planar indexing, and cell extraction, if desired.
        
        Lens correction is performed if n and f are provided.
        n and f can be found with the GetLensCorrectParams function.
        Cells are extracted if numCols and numRows are provided.

    Args:
        imagepath (str): path to a raw image
        savepath (str): folder path for saving output
        numCols (int): number of cells across in module image
        numRows (int): number of cells down in module image
        savesmall (bool): Save a smaller .jpg version of the planar indexed
            image with True. Default is False.
        imgtype (str):  'UVF' UV Fluorescence Image
                        'gradient' for unequal intensity across the image
                        'lowcon' low contrast between cell and background
                            or PL images without background subtraction
    Returns:
        
    """ 
    img = cv2.imread(imagepath)
    if (n is not None) and (f is not None):
        img = process.lensCorrect(img,n,f)
    planarindexed = process.PlanarIndex(img,imgtype)
    file_name = os.path.split(imagepath)[1]
    image_name = savepath+file_name
    cv2.imwrite(image_name,planarindexed)
    if savesmall == True:
        dims = planarindexed.shape
        y = int(dims[0]/3)
        x = int(dims[1]/3)
        jpg_name = os.path.splitext(image_name)[0]+'.jpg'
        cv2.imwrite(jpg_name,cv2.resize(planarindexed,(x,y)))
    if (numCols is not None) and (numRows is not None):
        cellarrays = process.CellExtract(planarindexed,numCols,numRows)
        for i in range(len(cellarrays)):
            out = cellarrays[i]
            file_name,ext = os.path.splitext(os.path.split(imagepath)[1])
            image_name = savepath+os.path.splitext(file_name)[0]+'-c'+ '{:02}'.format(i+1) + ext
            cv2.imwrite(image_name,out)
    return True

def RegisteredMMPipeline(input_path = None, extracted_path = ".", 
                         best_postfix = "_EL_9", folders =  None,
                         extension = ".tiff", cell_row = 2,
                         cell_col = 2, ignore_failure = True, out_shape = None):
    """
    Parameters
    ----------
    input_path : string, optional
        The default is None.  The path to the input data, which should be the output of process.CleanRawData
    extracted_path : string, optional
        The default is "." Where the folders of data should be output
    best_postfix : string, optional
        The default is "_EL_9". The postfix of the exposure/type (Usually long exposure EL) to be used for extracting all other exposures
    folders : list, optional
        The default is None. A list of paths to folders containing samples
    extension : string, optional
        The default is ".tiff". The image file extension
    cell_row : int, optional
        The default is 2. Number of cells in the horizontal direction
    cell_col : TYPE, optional
        The default is 2.
    ignore_failure : int, optional
        The default is True. Number of cells in the vertical direction

    Raises
    ------
    Exception
    Requires either a path of folders or a list of folder names.

    Returns
    -------
    failures : list
        A list of samples the pipeline failed on

    """
    failures = []
    if folders is None:
        folders = sorted(glob.glob("{input_path}*".format(input_path=input_path)))
    if folders is None and input_path is None:
        raise Exception("Require either base folder directory or list of sample folders")
    
    for sample_folder in folders:
        sa_number = sample_folder.split("/")[-1]
        
        best_sample = io.imread(sample_folder+"/" +sa_number+"{best_postfix}.tiff".format(best_postfix=best_postfix))
        
        other_images = []
        for filename in os.listdir(sample_folder):
            img = io.imread(sample_folder+ "/" + filename)
            postfix = filename[filename.find("_"):].split(extension)[0]
            if img is not None:
                other_images.append((postfix, img))
                
        cellarrays, col_ch_pts, row_ch_pts = process.CellExtract(best_sample,
                                                                 cell_row,cell_col, 
                                                                 changepoints = True)
        extracted_arrays = []
        for postfix, img in other_images:
            arrays = process.CellExtractByChangePoint(img, col_ch_pts, 
                                                      row_ch_pts, 
                                                      cell_row,cell_col)
            extracted_arrays.append((postfix,arrays))
        #del other_images
        M = None
        xdim = ydim = 0    
        for i in range(0, len(cellarrays)):
            flag =  True
            try:
                pi, M , xdim, ydim = process.PlanarIndex(cellarrays[i], ret_mask = True)
            except:
                print("Sample {sa_number} Cell {i} failed on postfix {best_postfix}".format(sa_number = sa_number, i=i, best_postfix=best_postfix))
                failures.append(sample_folder)
                pi = np.full(cellarrays[i].shape, np.median(cellarrays[i]), dtype = np.uint16)
                flag = False
                if not ignore_failure:
                    continue
            for postfix, arrays in extracted_arrays:
                if not flag:
                    cell = np.full(cellarrays[i].shape, np.median(arrays[i]))
                else:
                    cell = process.PlanarIndexByMask(arrays[i], mask = M, xdim = xdim, ydim = ydim)
                if not os.path.exists(extracted_path + sa_number+ "/"):
                    os.makedirs(extracted_path + sa_number + "/")
                if out_shape is not None:
                    cell = transform.resize(cell, out_shape)
                    pi = transform.resize(pi, out_shape)
                io.imsave(extracted_path + sa_number+ "/" + sa_number + postfix +'-c'+ '{:02}'.format(i+1) +'.tiff',cell,check_contrast=False)
                io.imsave(extracted_path + sa_number+ "/" + sa_number + best_postfix +'-c'+ '{:02}'.format(i+1) +'.tiff',pi,check_contrast=False)
    return list(set(failures)) # ensure uniqueness

def backgroundSubtract(postfix = 'EL_9', input_path = "." ):
    """
    Gievn a input folder and postfix, performs background subtraction

    Parameters
    ----------
    postfix : str, optional
        The default is 'EL_9'.
    input_path : str, optional
        The path to the img folder. The default is ".".

    Returns
    -------
    Numpy array
        The subtracted image.

    """
    files = glob.glob(input_path + "/*.tiff") 
    imgs_path = sorted([file for file in files if postfix in file])
    imgs = [io.imread(img) for img in imgs_path]
    return imgs[0]-imgs[1]
