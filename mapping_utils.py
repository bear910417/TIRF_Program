import h5py
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import bm3d
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.exposure import rescale_intensity
import os
import lmfit
from tqdm import tqdm
import scipy.ndimage
import time
from PIL import Image

class Glimpse_mapping:
    
    def __init__(self, path):
        
        self.path = path
        self.path_g = path + r'\\g'
        self.path_r = path + r'\\r'
        self.path_b = path + r'\\b'
    
    # Main function to process the image and obtain coordinates of blobs
    def map(self, mode, seg):
        
        # Load image
        self.seg = seg
        path_g = self.path_g
        path_r = self.path_r
        path_b = self.path_b
        
        # Select the channel
        if mode == 'g':
            path = path_g
            sw = 1
        elif mode == 'r':
            path = path_r
            sw = 0
        else:
            path = path_b
            sw = 2
            
        plot = False
        tic = time.perf_counter()
        
        file = h5py.File(path+r'\header.mat','r')
        nframes=int(file[r'/vid/nframes'][0][0])
        width=int(file[r'/vid/width/'][0][0])
        height=int(file[r'/vid/height/'][0][0])

        filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
        offset=file[r'/vid/offset'][:].flatten().astype('int')

        frame=np.zeros((height,width), dtype= np.int16)
        ave_arr = np.zeros((height,width), dtype= np.float32)
        nframes = 10
            
        gfilename = str(filenumber[0]) + '.glimpse'
        gfile_path = path+r'\\'+gfilename
        image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))

        try:
            gfilename = str(1) + '.glimpse'
            gfile_path = path+r'\\'+gfilename
            image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (height,width)))
            image_g = np.concatenate((image_g,image_g_1))
        except:
            pass
        image_g = image_g + 2**16

        

        # Average the image to obtain a representive image used for blob finding
        for j in range(self.seg * 10, self.seg * 10 + nframes):
            ave_arr= ave_arr + image_g[j]

        ave_arr = ave_arr/(nframes)
        frame = ave_arr

        # Rescale the image
        maxf=np.max(frame)
        minf=np.min(frame)

        frame=rescale_intensity(frame,in_range=(minf,maxf),out_range=np.ubyte)

        # Image denoising
        dframe= bm3d.bm3d(frame, 6, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        temp1=dframe


        # Select the part of the channel
        left_image  = temp1[0:height, 170 * sw + sw : 170 * sw + 170]
        left_image1  = frame[0:height, 170 * sw + sw :  170 * sw + 170]


        # Blob finding by DOG
        blobs_dog = blob_dog(left_image, min_sigma=1.5/sqrt(2),max_sigma=3.5/sqrt(2), threshold=3 ,overlap=0)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        blobs_dog=blobs_dog[blobs_dog[:,1].squeeze()<165]

        # Plot and save the processed image
        cpath=os.path.join(path,r'circled')
        if not os.path.exists(cpath):
            os.makedirs(cpath)

        im = Image.fromarray(left_image1)
        im.save(cpath+f'\\circled_{mode}.tif')
        self.left_image = left_image
        
        return blobs_dog
    
    # Function to get the processed image
    def get_image(self):
        return self.left_image