import os
from PIL import Image
from glob import glob
import numpy as np
import cv2
import torch
from scipy import signal

"""Below function converts RGB to YIQ format, in this to magnify phase we only magnify Y(luminance), to
   avoid color flickering effects"""
def rgb2yiq(rgb):
    #Computing luma channel Y
    y = rgb @ np.array([[0.30], [0.59], [0.11]])
    #Conver tot RBY format
    rby = rgb[:, :, (0,2)] - y
    #i,q form chrominance 
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    #final yiq output
    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq

"""Open CV loads image in bgr format, so we need separate function for that"""
def bgr2yiq(bgr):
    
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq

"""This function will convert our final processed image back to RGB"""
def yiq2rgb(yiq):
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb

"""Reads all frames from video files, rescales them, then converts them to YIQ"""
def get_video(video_path, scale_factor, colorspace_func=lambda x: x):
    
    frames = [] 
    cap = cv2.VideoCapture(video_path)

    #sampling rate
    fs = cap.get(cv2.CAP_PROP_FPS)

    idx = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == 0:
            og_h, og_w, _ = frame.shape
            w = int(og_w*scale_factor)
            h = int(og_h*scale_factor)

        frame = colorspace_func(np.float32(frame/255))
        frames.append(cv2.resize(frame, (w, h)))

        idx += 1
        
    cap.release()
    cv2.destroyAllWindows()
    del cap

    return frames, fs

"""Creates gif from images"""
def create_gif_from_images(save_path, image_path, ext):

    image_paths = sorted(glob(os.path.join(image_path, f'*.{ext}')))
    pil_images = [Image.open(im_path ) for im_path in image_paths]
    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=45, loop=0)
    
"""Gif function for numpy iamges"""
def create_gif_from_numpy(save_path, images):
  
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=45, loop=0)
    
"""Calculates FFT of batch of tensors (pyramid levels)"""
def get_fft2_batch(tensor_in):
    return torch.fft.fftshift(torch.fft.fft2(tensor_in, dim=(1,2))).type(torch.complex64)

"""To find FFT transfer function for a given bandpass filter, device here means CUDA or CPU"""
def bandpass_filter(freq_lo, freq_hi, fs, num_taps, device):
   
    freq_lo = freq_lo / fs * 2
    freq_hi = freq_hi / fs * 2

    bandpass = signal.firwin(numtaps=num_taps, cutoff=[freq_lo, freq_hi], pass_zero=False)
    
    bandpass = torch.tensor(bandpass).to(device)
    transfer_function = torch.fft.fft(torch.fft.ifftshift(bandpass)).type(torch.complex64)
    transfer_function = torch.tile(transfer_function, [1, 1, 1, 1]).permute(0, 3, 1, 2)

    return transfer_function


