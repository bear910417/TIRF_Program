import h5py
import numpy as np
import matplotlib.pyplot as plt
import bm3d
import time
from skimage.feature import blob_dog
from math import sqrt
from skimage.exposure import rescale_intensity
import os
import lmfit
from tqdm import tqdm
import scipy.ndimage
import cv2
from Blob import Blob



class Image_Loader():
    
    def __init__(self, n_pro, thres, path, g_length, r_length, b_length, g_start, r_start, b_start, bac_mode):
        
        self.thres = thres
        self.path = path
        self.path_g = path+r'\g'
        self.path_b = path+r'\b'     
        self.path_r = path+r'\r'
        self.g_length = g_length 
        self.r_length = r_length 
        self.b_length = b_length 
        self.g_start = int(g_start)
        self.r_start = int(r_start)
        self.b_start = int(b_start)
        self.n_pro = n_pro
        self.bac_mode = bac_mode
        self.tic = None
        self.dcombined_image = None
        self.dframe = None
        self.image_g = None
        self.image_r = None
        self.image_b = None
        self.M = None
        self.M_b = None
        self.b_exists = None
        self.r_exists = None
        self.bac_b = None
        self.bac = None
        self.cpath = None
        
        
    def gaussian_peaks(self, offy, offx):
        gaussian_filter = np.zeros((7,7), dtype=np.float32)
        offy = np.round(offy, 2)
        offx = np.round(offx, 2)

        for i in range (-3, 4): 
                for j in range (-3, 4):
                    dist = 0.3 * ((i - offy)**2 + (j- offx)**2)
                    gaussian_filter[i+3][j+3] = np.exp(-dist)
        return gaussian_filter
    
    def cal_bac(self, image, nframes, q = 0.5):
        bac = np.zeros((nframes, image.shape[1], image.shape[2]))
        for bt in range(0,nframes):
                bac_temp = image[bt]
                bw = 16 
                aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
                
                for i in range(0,self.height, bw): #0~480
                    for j in range(0,self.width, bw): #0~480
                            aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw], 0.5),1)

                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac[bt] = aves

        return np.average(bac, axis=0)


    def affine(self, x,y,M):
        x1 = M[0][0] * x + M[0][1] * y + M[0][2]
        y1 = M[1][0] * x + M[1][1] * y + M[1][2]
        
        return [y1, x1]
    
    def plot_circled(self, blobs_dog):
        fig = plt.figure()   
        ax = fig.add_subplot()
        ax.imshow(self.dcombined_image,cmap='Greys_r')
        
        for blob in blobs_dog:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='white', linewidth=0.5, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()

        plt.tight_layout()
        
        
        self.cpath = os.path.join(self.path,r'circled')
        os.makedirs(self.cpath, exist_ok = True)
        plt.savefig(self.cpath+f'\\circle_{self.n_pro}.tif',dpi=300)
        plt.close()

    def expand_bac(self, bac, length):
        bac = np.expand_dims(bac, 0)
        bac = np.repeat(bac, length, axis = 0)
        return bac
    

    def cal_time_g(self, path, start, length):
        file = h5py.File(path +r'\header.mat','r')
        time = np.array(file[r'/vid/ttb'][:]).reshape(-1).astype(float)
        time_n = np.cumsum(np.diff(np.concatenate(([time[0]], time))))*0.001
        time_n = time_n[start:start+length]

        return time_n, time[start]
    
    def cal_time(self, path, start, length, first):
        file = h5py.File(path +r'\header.mat','r')
        time = np.array(file[r'/vid/ttb'][:]).reshape(-1).astype(float)
        time_n = np.cumsum(np.diff(np.concatenate(([first], time))))*0.001
        time_n = time_n[start:start+length]

        return time_n   
    
    def load_image(self, fsc = None):
        
        self.tic = time.perf_counter()
        
        path = self.path
        path_g = self.path_g
        path_r = self.path_r
        path_b = self.path_b
        print(r'processing channel:')
        if os.path.exists(path_g) == True:
            g_exists = 1
            print(r'green')
        else:
            g_exists = 0

        if os.path.exists(path_b) == True:
            b_exists = 1
            print(r'blue')
        else:
            b_exists = 0

        if os.path.exists(path_r) == True:
            r_exists = 1
            print(r'red')
        else:
            r_exists = 0
            
        self.r_exists = r_exists
        self.b_exists = b_exists
        self.g_exists = g_exists
        bac_mode = self.bac_mode 
         

        
        nframes_true = 0
        self.width = 512
        self.height = 512
        filenumber = [0]
        first = None

        #g_exist?  
        time_g = np.zeros(10)
        image_g = np.zeros((10, 512, 512))  
        if  g_exists == 1:
            
            file = h5py.File(path_g+r'\header.mat','r')
            nframes_true = int(file[r'/vid/nframes'][0][0])
            self.width=int(file[r'/vid/width/'][0][0])
            self.height=int(file[r'/vid/height/'][0][0])
            filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
            gfilename = str(filenumber[0]) + '.glimpse'
            gfile_path = path_g+r'\\'+gfilename
            time_g, first = self.cal_time_g(path_g, self.g_start, self.g_length)
            image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
            nframes = min(20, nframes_true)  

            for n in range (1, 10):
                try:
                    gfilename = str(n) + '.glimpse'
                    gfile_path = path_g+r'\\'+gfilename
                    image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                    image_g = np.concatenate((image_g, image_g_1))
                except:
                    pass
    
            image_g = image_g + 2**15
            print(f'Calculating g Backgrounds with mode {bac_mode}') 
            self.bac_g = self.cal_bac(image_g, nframes, 0.5) 
            try:
                fsc.set("load_progress", '0')
            except:
                pass

        
        #r_exist?
        time_r = np.zeros(10)
        image_r = np.zeros((10, 512, 512))

        if  r_exists == 1:
            file = h5py.File(path_r+r'\header.mat','r')
            rfilename = str(filenumber[0]) + '.glimpse'
            rfile_path = path_r+r'\\'+rfilename
            nframes_true = int(file[r'/vid/nframes'][0][0])
            if first == None:
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_r, first = self.cal_time_g(path_r, self.r_start, self.r_length)
            else:
                time_r = self.cal_time(path_r, self.r_start, self.r_length, first)
            image_r = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
            nframes = min(20, nframes_true)  
            for n in range (1, 10):
                try:
                    rfilename = str(n) + '.glimpse'
                    rfile_path = path_r + r'\\'+rfilename
                    image_r_1 = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                    image_r = np.concatenate((image_r, image_r_1))
                except:
                    pass
            image_r = image_r + 2**15

            print(f'Calculating r Backgrounds with mode {bac_mode}')  
            self.bac_r = self.cal_bac(image_r, nframes, 0.5) 
            try:
                fsc.set("load_progress", '0')
            except:
                pass
        
            
        #b_exist?    
        time_b = np.zeros(10)
        image_b = np.zeros((10, 512, 512))

        if  b_exists == 1:
            print(f'Calculating b Backgrounds with mode {bac_mode}')  
            file = h5py.File(path_b+r'\header.mat','r')
            bfilename = str(filenumber[0]) + '.glimpse'
            bfile_path = path_b+r'\\'+bfilename
            nframes_true = int(file[r'/vid/nframes'][0][0])
            if first == None:   
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_b, first = self.cal_time_g(path_b, self.b_start, self.b_length)
            else:
                time_b = self.cal_time(path_b, self.b_start, self.b_length, first)
            image_b = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height, self.width)))
            nframes = min(20, nframes_true)  
            
            for n in range (1, 10):
                try:
                    bfilename = str(n) + '.glimpse'
                    bfile_path = path_b+r'\\'+bfilename
                    image_b_1 = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                    image_b = np.concatenate((image_b, image_b_1))
                except:
                    pass


            image_b = image_b + 2**15      
            self.bac_b = self.cal_bac(image_b, nframes, 0.5) 
        ###



        #rescale image intensity and remove background

        self.image_g = image_g
        self.image_r = image_r
        self.image_b = image_b
        self.b_exists = b_exists
        self.g_exists = g_exists
        self.r_exists = r_exists
        
        return  time_g, time_r, time_b, nframes_true
    
    
    
    def gen_dimg(self, anchor, mpath, maxf = 420, minf = 178, channel = 'green', plot = True, ch = 'all'):
        
        dframe_g = 0
        dframe_b = 0
        dframe_r = 0
        nframes = 20

        if  (self.r_exists == 1 and ch == 'all') or ch == 'red':
            end = min(self.image_r.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            frame_r = np.average(self.image_r[start:end], axis = 0)
            frame_r = rescale_intensity(frame_r,in_range = (minf,maxf), out_range=np.ubyte)
            dframe_r = bm3d.bm3d(frame_r, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)
        
        if  (self.g_exists == 1 and ch == 'all') or ch == 'green':
            end = min(self.image_g.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            frame_g = np.average(self.image_g[start:end], axis = 0)
            frame_g = rescale_intensity(frame_g, in_range = (minf,maxf), out_range = np.ubyte)
            dframe_g = bm3d.bm3d(frame_g, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)


        if  (self.b_exists == 1 and ch == 'all') or ch == 'blue':
            end = min(self.image_b.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            frame_b = np.average(self.image_b[start:end], axis = 0)
            frame_b = rescale_intensity(frame_b,in_range = (minf,maxf), out_range=np.ubyte)
            dframe_b = bm3d.bm3d(frame_b, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)



        ch_dict = {'green' :  dframe_g,
                    'blue' :  dframe_b,
                    'red' :  dframe_r
                    }


        dframe = ch_dict[channel] 

        if ch == 'all':
            #combine two channel image
            self.M = np.load(mpath + r'\map_g_r.npy')
            self.Mb = np.load(mpath + r'\map_g_b.npy')



            left_image  = dframe[0:self.height,0:170]
            right_image = dframe[0:self.height,171:341]
            blue_image = dframe[0:self.height,342:512]
            rows, cols = right_image.shape
            
            left_image_trans = cv2.warpAffine(left_image, self.M, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
            blue_image_trans = cv2.warpAffine(blue_image, self.Mb, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
                
            dcombined_image = (right_image + left_image_trans + blue_image_trans)
            toc = time.perf_counter()
            print(f"Finished in {toc - self.tic:0.4f} seconds")
            
            
            self.dcombined_image = dcombined_image
        
        self.dframe = dframe

        if self.b_exists:
            self.dframe_b = dframe_b
        else:
            self.dframe_b = dframe

        if self.g_exists:
            self.dframe_g = dframe_g
        else:
            self.dframe_g = dframe
        
        if self.r_exists:
            self.dframe_r = dframe_r
        else:
            self.dframe_r = dframe
        
        if channel == 'red':
            return self.dframe_r 
        elif channel == 'green':
            return self.dframe_g 
        elif channel == 'blue':
            return self.dframe_b 
        else:
            return None

    
    
    def det_blob(self, plot = False, fsc = None, thres = None, r = 3, redchi_thres = 400):
        if thres != None:
            self.thres = thres

        print('Finding blobs')      
        blobs_dog = blob_dog(self.dcombined_image, min_sigma= (r-1) /sqrt(2), max_sigma = r /sqrt(2), threshold=self.thres, overlap=0, exclude_border = 2)
        print(f'Found {blobs_dog.shape[0]} preliminary blobs')

        if plot == True:
            self.plot_circled(blobs_dog)

        params = lmfit.Parameters()
        params.add('centery', value = 3, min = 2, max = 4)
        params.add('centerx', value = 3, min = 2, max = 4)
        params.add('amplitude', value = 5000)
        params.add('sigmay', value = 3, min = 0, max = 6)
        params.add('sigmax', value = 3, min = 0, max = 6)
       
        coord_list = []
        blob_list = []



        for i, raw_blob in enumerate(tqdm(blobs_dog)):

            try:
                fsc.set("progress", str(i / (blobs_dog.shape[0]-1)))
            except:
                pass
            
            b = Blob(raw_blob, self.M, self.Mb)
            b.map_coord()
            b.check_bound()

            b.set_image(self.dframe_r, 'red')
            b.set_image(self.dframe_g, 'green')
            b.set_image(self.dframe_b, 'blue')

            b.check_max(self.dcombined_image)



            if self.r_exists or self.g_exists or self.b_exists:
                b.gaussian_fit(0)
            
            if self.g_exists or self.b_exists:
                b.gaussian_fit(1)

            if self.b_exists:
                b.gaussian_fit(2)

            b.check_fit(redchi_thres)

            if b.quality == 1:
                coord_list.append(b.get_coord())
                blob_list.append(b)
                if plot == True:
                    b.plot_circle(self, self.dframe_g, self.dframe_b, i)
            
            

        self.blob_list = blob_list
        print(f'Found {len(coord_list)} filterd blobs')
        return coord_list
    
    def cal_drift(self, anchor, channel):
        image = self.gen_dimg(anchor, mpath = None, maxf = 420, minf = 178, channel = 'green', plot = False, ch = channel)
        coord_list_drift = []
        channel_dict = {
            'red' : 0,
            'green' : 1,
            'blue' : 2
        }

        for b in self.blob_list:
            b.set_image(image = image, channel = channel)
            b.set_params(channel_dict[channel])
            b.check_bound()
            b.gaussian_fit(channel_dict[channel], nfev = 10)       
            coord_list_drift.append(b.get_coord())

        return coord_list_drift


    def cal_intensity(self, coord_list, maxf = 35000, minf = 32946, fsc = None):
        
        print('Calcultating Intensities')
        i=0

        trace_gg = np.zeros((1000,int(self.g_length)))
        trace_gr = np.zeros((1000,int(self.g_length)))
        trace_rr = np.zeros((1000,int(self.r_length)))
        trace_bb = np.zeros((1000,int(self.b_length)))
        trace_bg = np.zeros((1000,int(self.b_length)))
        trace_br = np.zeros((1000,int(self.b_length)))
        total_blobs = len(coord_list)
        b_snap = np.zeros((total_blobs, 3, self.b_length, 9, 9))
        g_snap = np.zeros((total_blobs, 2, self.g_length, 9, 9))
        r_snap = np.zeros((total_blobs, 1, self.r_length, 9, 9))


        self.cpath=os.path.join(self.path,r'circled')
        os.makedirs(self.cpath+f'\\{self.n_pro}', exist_ok=True)
        for blob_count, blob in enumerate(coord_list):

            try:
                fsc.set("cal_progress", str(blob_count / (len(coord_list)-1)))
            except:
                pass

            yr, xr, yg, xg, yb, xb, ymr, xmr, ymg, xmg, ymb, xmb = blob
            r = 3

            
            yr = int(yr)
            xr = int(xr)
            yg = int(yg)
            xg = int(xg)
            yb = int(yb)
            xb = int(xb)


            if self.r_exists ==1:
                srr = self.gaussian_peaks(ymr, xmr)
                bac_r = self.expand_bac(self.bac_r, self.r_length)
                trace_rr[i] = np.sum(2 * srr *(self.image_r[:, yr-r:yr+r+1,xr-r:xr+r+1] - bac_r[:, yr-r:yr+r+1,xr-r:xr+r+1]), axis = (1, 2))
                r_snap[blob_count][0] = self.image_r[:, yr-4:yr+4+1,xr-4:xr+4+1]


            if self.g_exists ==1:
                sgg = self.gaussian_peaks(ymg, xmg)
                sgr = self.gaussian_peaks(ymr, xmr)
                bac_g = self.expand_bac(self.bac_g, self.g_length)

                trace_gg[i] = np.sum(2 * sgg *(self.image_g[:, yg-r:yg+r+1,xg-r:xg+r+1] - bac_g[:, yg-r:yg+r+1,xg-r:xg+r+1]), axis = (1, 2))
                g_snap[blob_count][0] = self.image_g[:, yg-4:yg+4+1,xg-4:xg+4+1]

                trace_gr[i] = np.sum(2 * sgr *(self.image_g[:, yr-r:yr+r+1,xr-r:xr+r+1]- bac_g[:, yr-r:yr+r+1,xr-r:xr+r+1]), axis = (1, 2))
                g_snap[blob_count][1] = self.image_g[:, yr-4:yr+4+1,xr-4:xr+4+1]

                    
              
            if  self.b_exists == 1:
                sbb = self.gaussian_peaks(ymb, xmb)
                sbg = self.gaussian_peaks(ymg, xmg)
                sbr = self.gaussian_peaks(ymr, xmr)

                bac_b = self.expand_bac(self.bac_b, self.b_length)
                trace_bb[i] = np.sum(2 * sbb *(self.image_b[:, yb-r:yb+r+1,xb-r:xb+r+1] - bac_b[:, yb-r:yb+r+1,xb-r:xb+r+1]), axis = (1, 2))
                b_snap[blob_count][0] = self.image_b[:, yb-4:yb+4+1,xb-4:xb+4+1]

                trace_bg[i] = np.sum(2 * sbg *(self.image_b[:, yg-r:yg+r+1,xg-r:xg+r+1] - bac_b[:, yg-r:yg+r+1,xg-r:xg+r+1]), axis = (1, 2))
                b_snap[blob_count][1] = self.image_b[:, yg-4:yg+4+1,xg-4:xg+4+1]

                trace_br[i] = np.sum(2 * sbr *(self.image_b[:, yr-r:yr+r+1,xr-r:xr+r+1] - bac_b[:, yr-r:yr+r+1,xr-r:xr+r+1]), axis = (1, 2))
                b_snap[blob_count][2] = self.image_b[:, yr-4:yr+4+1,xr-4:xr+4+1]

            i = i+1
             
        trace_gg = trace_gg[0:i]
        trace_gr = trace_gr[0:i]
        trace_rr = trace_rr[0:i]
        trace_bb = trace_bb[0:i]
        trace_bg = trace_bg[0:i]
        trace_br = trace_br[0:i]
        
        np.savez(self.path + r'\blobs.npz', b = b_snap, g = g_snap, r = r_snap, minf = minf, maxf = maxf)
        
        return trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i
        
        
