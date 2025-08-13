"""Solar cell image analysis submodule.

This module provides functions for analyzing solar cell images, including:
- Busbar detection and removal
- Busbar width measurement
- Corner triangle detection

The module is designed for processing solar cell wafer images to extract
geometric features and remove artifacts like busbars for further analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import filters, morphology, measure
import cv2
import glob
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Union
import warnings
from tqdm import tqdm


def detect_busbars(image: np.ndarray, smoothing_sigma: float = 2, expected_count: int = 12, 
                  busbar_width: int = 4) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect busbars in a solar cell image using intensity analysis.
    
    This function analyzes the horizontal intensity profile of a solar cell image
    to detect busbar locations. Busbars appear as dark horizontal lines across
    the cell surface. The detection uses signal processing techniques with
    spacing constraints based on expected busbar count.
    
    Args:
        image (np.ndarray): Input solar cell image. Can be RGB (H, W, 3) or 
            grayscale (H, W). Will be converted to grayscale internally.
        smoothing_sigma (float, optional): Standard deviation for Gaussian 
            smoothing of the intensity profile. Higher values provide more 
            smoothing. Defaults to 2.
        expected_count (int, optional): Expected number of busbars in the image. 
            Used to constrain detection and filter results. Defaults to 12.
        busbar_width (int, optional): Width of each busbar in pixels for mask 
            creation. Defaults to 4.
    
    Returns:
        tuple: A 6-element tuple containing:
            - mask (np.ndarray): Boolean mask indicating busbar locations 
              (True = busbar pixel)
            - coverage (float): Percentage of image area covered by detected 
              busbars (0-100)
            - row_sums (np.ndarray): Original row-wise mean intensity values
            - row_sums_smooth (np.ndarray): Smoothed row-wise mean intensity values
            - inverted_signal (np.ndarray): Inverted and normalized intensity 
              signal used for peak detection
            - peaks (np.ndarray): Array of row indices where busbars were detected
    
    Note:
        The function uses adaptive thresholding - if the expected number of
        busbars is not found with initial parameters, it adjusts the detection
        sensitivity automatically.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    h, w = gray.shape
    
    # Sum each row
    row_sums = np.mean(gray, axis=1)
    
    # Smooth the signal
    row_sums_smooth = ndimage.gaussian_filter1d(row_sums, sigma=smoothing_sigma)
    
    # Invert signal for peak detection
    inverted_signal = 1 - row_sums_smooth
    inverted_signal = (inverted_signal - inverted_signal.min()) / (inverted_signal.max() - inverted_signal.min())
    
    # Estimate expected spacing
    expected_spacing = h / (expected_count + 1)  # +1 because busbars don't go to edges
    min_distance = int(expected_spacing * 0.6)  # Allow some variation
    
    # Find peaks with spacing constraint
    peaks, properties = signal.find_peaks(inverted_signal, 
                                        height=0.2,  # Lower threshold initially
                                        prominence=0.01,
                                        distance=min_distance)
    
    # If we found too many peaks, keep only the strongest ones
    if len(peaks) > expected_count:
        # Sort by peak height and keep the strongest ones
        peak_heights = inverted_signal[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Descending order
        best_peaks = peaks[sorted_indices[:expected_count]]
        best_peaks = np.sort(best_peaks)  # Sort by position
        peaks = best_peaks
    
    # If we found too few peaks, lower the threshold
    elif len(peaks) < expected_count:
        peaks, properties = signal.find_peaks(inverted_signal, 
                                            height=0.1,  # Lower threshold
                                            prominence=0.005,
                                            distance=min_distance)
        if len(peaks) > expected_count:
            peak_heights = inverted_signal[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]
            best_peaks = peaks[sorted_indices[:expected_count]]
            best_peaks = np.sort(best_peaks)
            peaks = best_peaks
    
    # Create mask
    mask = np.zeros_like(gray, dtype=bool)
    for peak in peaks:
        start_row = max(0, peak - busbar_width // 2)
        end_row = min(h, peak + busbar_width // 2 + 1)
        mask[start_row:end_row, :] = True
    
    coverage = np.sum(mask) / mask.size * 100
    
    return mask, coverage, row_sums, row_sums_smooth, inverted_signal, peaks
    

def remove_busbars_with_inpainting(image: np.ndarray, busbar_positions: np.ndarray, 
                                 busbar_width: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Remove busbars from solar cell image using OpenCV inpainting.
    
    This function removes detected busbars by inpainting the busbar regions
    with synthesized texture that matches the surrounding cell surface. The
    inpainting process uses the TELEA algorithm which is effective for thin
    structures like busbars.
    
    Args:
        image (np.ndarray): Input solar cell image. Can be RGB (H, W, 3) or 
            grayscale (H, W). Will be converted to grayscale internally.
        busbar_positions (np.ndarray): Array of row indices indicating the 
            center positions of busbars to be removed.
        busbar_width (int, optional): Width of busbars in pixels. This determines
            how many rows around each center position will be inpainted. 
            Defaults to 4.
    
    Returns:
        tuple: A 2-element tuple containing:
            - result_image (np.ndarray): Image with busbars removed through 
              inpainting, same type and range as input
            - mask (np.ndarray): Binary mask (uint8) showing inpainted regions 
              (255 = inpainted, 0 = original)
    
    Note:
        The function handles data type conversions automatically - if the input
        is not uint8, it converts to uint8 for OpenCV processing and then
        converts back to the original data type and range.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Ensure proper data types for OpenCV
    if gray.dtype != np.uint8:
        # Convert to uint8 if needed
        gray_normalized = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
    else:
        gray_normalized = gray
    
    # Create mask for busbar regions
    mask = np.zeros(gray_normalized.shape, dtype=np.uint8)
    h, w = gray_normalized.shape
    
    for busbar_center in busbar_positions:
        start_row = max(0, busbar_center - busbar_width // 2)
        end_row = min(h, busbar_center + busbar_width // 2 + 1)
        mask[start_row:end_row, :] = 255
    
    # Use OpenCV inpainting
    result_image = cv2.inpaint(gray_normalized, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Convert back to original data type if needed
    if gray.dtype != np.uint8:
        result_image = result_image.astype(np.float64) / 255.0 * (gray.max() - gray.min()) + gray.min()
        result_image = result_image.astype(gray.dtype)
    
    return result_image, mask


def measure_busbar_widths(image: np.ndarray, busbar_positions: np.ndarray, 
                         cell_size_mm: float = 182, smoothing_sigma: float = 2) -> Tuple[List[float], List[float]]:
    """Measure the physical width of detected busbars in pixels and millimeters.
    
    This function analyzes the intensity profile around each detected busbar
    to measure its actual width using the Full Width at Half Maximum (FWHM)
    method. The measurements are provided in both pixel units and physical
    units (millimeters) based on the cell size calibration.
    
    Args:
        image (np.ndarray): Input solar cell image. Can be RGB (H, W, 3) or 
            grayscale (H, W). Will be converted to grayscale internally.
        busbar_positions (np.ndarray): Array of row indices indicating the 
            center positions of busbars to measure.
        cell_size_mm (float, optional): Physical width of the solar cell in 
            millimeters. Used for pixel-to-mm conversion. Standard solar cells 
            are typically 182mm wide. Defaults to 182.
        smoothing_sigma (float, optional): Standard deviation for Gaussian 
            smoothing of the intensity profile before width measurement. 
            Defaults to 2.
    
    Returns:
        tuple: A 2-element tuple containing:
            - busbar_widths_pixels (List[float]): List of busbar widths in pixels,
              one measurement per busbar in busbar_positions
            - busbar_widths_mm (List[float]): List of busbar widths in millimeters,
              converted using the cell_size_mm calibration
    
    Note:
        The FWHM method finds the width of each busbar by locating where the
        intensity profile reaches half the maximum difference between the 
        busbar center (minimum) and the surrounding cell surface (baseline).
        If measurement fails for a busbar, a default width of 4 pixels is used.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    h, w = gray.shape
    
    # Calculate pixel to mm conversion
    pixels_per_mm = w / cell_size_mm
    
    # Get row intensity profile
    row_sums = np.mean(gray, axis=1)
    row_sums_smooth = ndimage.gaussian_filter1d(row_sums, sigma=smoothing_sigma)
    
    busbar_widths_pixels = []
    busbar_widths_mm = []
    
    # For each busbar, measure its width
    for i, busbar_center in enumerate(busbar_positions):
        # Define search window around busbar center
        search_half_width = 15  # pixels to search on each side
        start_row = max(0, busbar_center - search_half_width)
        end_row = min(h, busbar_center + search_half_width + 1)
        
        # Extract intensity profile around this busbar
        local_profile = row_sums_smooth[start_row:end_row]
        local_rows = np.arange(start_row, end_row)
        
        # Find the minimum (darkest point - should be busbar center)
        local_min_idx = np.argmin(local_profile)
        
        # Method: Full Width at Half Maximum (FWHM)
        min_intensity = local_profile[local_min_idx]
        
        # Find baseline intensity (average of edges of search window)
        edge_intensities = [local_profile[0], local_profile[-1]]
        baseline = np.mean(edge_intensities)
        
        # Half maximum between min and baseline
        half_max = min_intensity + 0.5 * (baseline - min_intensity)
        
        # Find points where profile crosses half maximum
        left_edge = None
        right_edge = None
        
        # Search left from center
        for j in range(local_min_idx, -1, -1):
            if local_profile[j] > half_max:
                if j < local_min_idx:  # Make sure we moved away from center
                    left_edge = local_rows[j+1] if j+1 < len(local_rows) else local_rows[j]
                break
        
        # Search right from center
        for j in range(local_min_idx, len(local_profile)):
            if local_profile[j] > half_max:
                if j > local_min_idx:  # Make sure we moved away from center
                    right_edge = local_rows[j-1] if j-1 >= 0 else local_rows[j]
                break
        
        if left_edge is not None and right_edge is not None:
            width_fwhm_pixels = right_edge - left_edge
            width_fwhm_mm = width_fwhm_pixels / pixels_per_mm
        else:
            width_fwhm_pixels = 4.0  # fallback default
            width_fwhm_mm = width_fwhm_pixels / pixels_per_mm
        
        busbar_widths_pixels.append(width_fwhm_pixels)
        busbar_widths_mm.append(width_fwhm_mm)
    
    return busbar_widths_pixels, busbar_widths_mm


def detect_corner_triangles_shape_based(image: np.ndarray, corner_size_fraction: float = 0.25) -> Tuple[np.ndarray, Dict, int]:
    """Detect triangular corner regions in solar cell images using shape analysis.
    
    Solar cells often have triangular corners that are cut off or masked during
    manufacturing. This function detects these corner triangles by analyzing
    each corner region of the image for triangular dark shapes. The detection
    uses automatic thresholding and connected component analysis.
    
    Args:
        image (np.ndarray): Input solar cell image. Can be RGB (H, W, 3) or 
            grayscale (H, W). Will be converted to grayscale internally.
        corner_size_fraction (float, optional): Fraction of image dimensions to 
            use as the corner analysis region. For example, 0.25 means each 
            corner region will be 25% of the image height and width. Must be 
            between 0 and 1. Defaults to 0.25.
    
    Returns:
        tuple: A 3-element tuple containing:
            - corner_mask (np.ndarray): Boolean mask indicating all detected 
              corner triangle pixels (True = corner triangle pixel)
            - corner_stats (Dict): Dictionary with keys 'top_left', 'top_right', 
              'bottom_left', 'bottom_right'. Each value is a dict containing:
              * 'area_pixels' (int): Area of detected triangle in pixels
              * 'area_fraction' (float): Fraction of corner region occupied
              * 'centroid' (tuple or None): Center coordinates of triangle
              * 'bbox' (tuple or None): Bounding box coordinates
            - total_corner_area (int): Total number of pixels identified as 
              corner triangles across all corners
    
    Note:
        The function uses Otsu's automatic thresholding method to separate
        dark triangle regions from the cell surface. Only the largest connected
        component in each corner is considered as the triangle, filtering out
        noise and small artifacts.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float64)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    h, w = gray.shape
    
    corner_h = int(h * corner_size_fraction)
    corner_w = int(w * corner_size_fraction)
    
    corners = {
        'top_left': (slice(0, corner_h), slice(0, corner_w)),
        'top_right': (slice(0, corner_h), slice(w-corner_w, w)),
        'bottom_left': (slice(h-corner_h, h), slice(0, corner_w)),
        'bottom_right': (slice(h-corner_h, h), slice(w-corner_w, w))
    }
    
    corner_mask = np.zeros_like(gray, dtype=bool)
    corner_stats = {}
    
    for corner_name, (row_slice, col_slice) in corners.items():
        corner_region = gray[row_slice, col_slice]
        
        # Use Otsu's method for automatic thresholding
        from skimage.filters import threshold_otsu
        try:
            otsu_threshold = threshold_otsu(corner_region)
            corner_dark = corner_region < otsu_threshold
        except:
            # Fallback if Otsu fails
            corner_dark = corner_region < np.percentile(corner_region, 15)
        
        # Keep only the largest connected component (should be the triangle)
        labeled = measure.label(corner_dark)
        if labeled.max() > 0:
            # Find largest component
            props = measure.regionprops(labeled)
            largest_area = max(prop.area for prop in props)
            largest_prop = max(props, key=lambda x: x.area)
            
            # Create mask for just the largest component
            largest_mask = labeled == largest_prop.label
            
            # Map back to full image coordinates
            corner_mask[row_slice, col_slice] |= largest_mask
            
            # Calculate statistics
            corner_stats[corner_name] = {
                'area_pixels': largest_area,
                'area_fraction': largest_area / (corner_h * corner_w),
                'centroid': largest_prop.centroid,
                'bbox': largest_prop.bbox
            }
        else:
            corner_stats[corner_name] = {
                'area_pixels': 0,
                'area_fraction': 0,
                'centroid': None,
                'bbox': None
            }
    
    total_corner_area = np.sum(corner_mask)
    
    return corner_mask, corner_stats, total_corner_area