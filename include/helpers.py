import os

import numpy as np
from skimage.io import imread

def load_img(path, img_name):
    img_path = os.path.join(path, img_name + ".png")
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img_clean = img / 255.
    return img_clean

def load_mask(path):
    img_path = os.path.join(path, "mask.png")
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img_clean = img / np.amax(img)
    return img_clean

def mse(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat - x_true))
    energy = np.mean(np.square(x_true))    
    return mse/energy

def psnr(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat - x_true))
    psnr_ = 10. * np.log(maxv** 2 /mse) / np.log(10.)
    return psnr_