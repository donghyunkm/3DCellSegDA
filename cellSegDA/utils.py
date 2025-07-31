# Util functions. Use these functions to generate masks and centers for EmbedSeg training data.
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tifffile import imsave, imwrite
import torch
import os
from skimage.color import rgb2gray
import cv2

def center_crop(img, dim):
    left = (img.shape[1] - dim) // 2
    right = (img.shape[1] - dim) // 2 + dim
    return img[:, left:right, left:right]

def crop_masks_and_images(paths, new_dir, newdim):
    for path in paths:
        image_array = tifffile.imread(path)
        image_array = center_crop(image_array, newdim)
        imwrite(os.path.join(new_dir, path.split("/")[-1]), image_array, imagej=True)

def normalize_images(img_path):
    img = imread(img_path)
    normalized_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = normalized_img.astype(np.uint8)
    return normalized_img

def crop_centers(center_paths, new_dir, newdim):
    for center in center_paths:
        image_array = tifffile.imread(center)
        image_array = center_crop(image_array, newdim)
        imwrite(os.path.join(new_dir, center.split("/")[-1]), image_array) #no imagej format because boolean format

def plot_each_2d_slice(image_path, mask_path):
    image = imread(image_path)
    mask = imread(mask_path)
                    
    for i in range(len(a)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image[i]) 
        axes[0].axis('off') 
        
        axes[1].imshow(mask[i])
        axes[1].axis('off')

        plt.show()