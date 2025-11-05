import numpy as np
import torch

## ==========================================================================================
"""Below function outputs a polar grid, which maps each x,y to r,theta polar coordinates"""
def get_polar_grid(h, w):
    h2 = h//2
    w2 = w//2

    #Computing normalised coordinates relative to center of image
    wx, wy = np.meshgrid(np.arange(-w2, w2 + (w % 2))/w2, 
                         np.arange(-h2, h2 + (h % 2))/h2)
    
    angle = np.arctan2(wy, wx)
    radius = np.sqrt(wx**2 + wy**2)
    #added to prevent edge cases where radius=0 at center
    radius[h2][w2] = radius[h2][w2 - 1]

    return angle, radius

"""This funtion is purel an optimisation, since FFT filters contains many zero values, we crop them so we
    have to deal with fewer values in array, store lesser pixels and do faster convolutions. 
    It out-puts the bounding-box for non-zero values"""
def get_filter_crops(filter_in):
    
    h, w = filter_in.shape
    above_zero = filter_in > 1e-10

    dim1 = np.sum(above_zero, axis=1)
    dim1 = np.where(dim1 > 0)[0]
    row_idx = np.clip([dim1.min() - 1, dim1.max() + 1], 0, h)

    dim2 = np.sum(above_zero, axis=0)
    dim2 = np.where(dim2 > 0)[0]
    col_idx = np.clip([dim2.min() - 1, dim2.max() + 1], 0, w)

    return np.concatenate((row_idx, col_idx))

"""This function just outputs the list cropped filters for a list of filters and crops"""
def get_cropped_filters(filters, crops):
    
    cropped_filters = []
    for (filt, crop) in zip(filters, crops):
        cropped_filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])

    return cropped_filters

"""This function just applies a filter in freq domain, then does IFFT to get band-passed image.
   It will be looped multiple times over various orientations for same scale to build levels of pyarmid."""
def build_level(image_dft, filt):
    return torch.fft.ifft2(torch.fft.ifftshift(image_dft * filt))

"""This will reconstruct spatial image which has undergone motion_magnification processing
   back into frequency domain so it can be used to calculate pyramid levels."""
def recon_level(pyr_level, filt):
    return 2.0 * torch.fft.fftshift(torch.fft.fft2(pyr_level)) * filt