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
        for i in range (-3, 4): 
                for j in range (-3, 4):
                    dist = 0.3 * ((i - offy)**2 + (j- offx)**2)
                    gaussian_filter[i+3][j+3] = np.exp(-dist)
        return gaussian_filter


        

    def affine(self, x,y,M):
        x1 = M[0][0] * x + M[0][1] * y + M[0][2]
        y1 = M[1][0] * x + M[1][1] * y + M[1][2]
        
        return [y1, x1]
        
    

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
         

        nframes = 10  
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
            nframes_true=int(file[r'/vid/nframes'][0][0])
            self.width=int(file[r'/vid/width/'][0][0])
            self.height=int(file[r'/vid/height/'][0][0])
            filenumber=file[r'/vid/filenumber/'][:].flatten().astype('int')
            gfilename = str(filenumber[0]) + '.glimpse'
            gfile_path = path_g+r'\\'+gfilename
            time_g, first = self.cal_time_g(path_g, self.g_start, self.g_length)
            image_g = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))

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
            bac=[]
            for bt in range(0,nframes):
                try:
                    fsc.set("load_progress", str(bt / (nframes-1) - 0.4))
                except:
                    pass
                
                bac_temp = image_g[bt]
                bw = 16 
                aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
                
                for i in range(0,self.height, bw): #0~480
                    for j in range(0,self.width, bw): #0~480
                        if bac_mode == 0:
                            aves[int((i-8)/16)][int((j-8)/16)] = np.round(np.amin(bac_temp[i:i+bw,j:j+bw]),1)
                        else:
                            aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw],0.4),1)

                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac.append(aves)

            self.bac_g = np.average(bac,axis=0)
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
            if first == None:
                nframes_true = int(file[r'/vid/nframes'][0][0])
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_r, first = self.cal_time_g(path_r, self.r_start, self.r_length)
            else:
                time_r = self.cal_time(path_r, self.r_start, self.r_length, first)
            image_r = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
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
            bac_r = []
            for bt in range(0,nframes):
                try:
                    fsc.set("load_progress", str(bt / (nframes-1) - 0.4))
                except:
                    pass
                
                bac_temp = image_r[bt]
                bw = 16 
                aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
                
                for i in range(0,self.height, bw): #0~480
                    for j in range(0,self.width, bw): #0~480
                        aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw],0.4),1)

                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac_r.append(aves)

            self.bac_r = np.average(bac_r, axis=0)
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

            if first == None:
                nframes_true = int(file[r'/vid/nframes'][0][0])
                self.width=int(file[r'/vid/width/'][0][0])
                self.height=int(file[r'/vid/height/'][0][0])
                time_b, first = self.cal_time_g(path_b, self.b_start, self.b_length)
            else:
                time_b = self.cal_time(path_b, self.b_start, self.b_length, first)
            image_b = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height, self.width)))
            
            for n in range (1, 10):
                try:
                    bfilename = str(n) + '.glimpse'
                    bfile_path = path_b+r'\\'+bfilename
                    image_b_1 = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                    image_b = np.concatenate((image_b, image_b_1))
                except:
                    pass


            image_b = image_b + 2**15
            
            bac_b = []
            for bt in range(0,nframes):
                try:
                    fsc.set("load_progress", str(bt / (nframes-1) - 0.4))
                except:
                    pass
                bac_temp = image_b[bt]
                bw = 16 
                aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
                for i in range(0,self.height, bw): #0~480
                    for j in range(0,self.width, bw): #0~480
                        aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw],0.5),1)

                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac_b.append(aves)   
                
            self.bac_b = np.average(bac_b,axis=0)
        ###



        #rescale image intensity and remove background

        self.image_g = image_g
        self.image_r = image_r
        self.image_b = image_b
        self.b_exists = b_exists
        self.g_exists = g_exists
        self.r_exists = r_exists
        
        return  time_g, time_r, time_b, nframes_true
    
    
    
    def gen_dimg(self, anchor, mpath, maxf = 35000, minf = 32946, channel = 'green', plot = True):
        

        ave_arr_g = np.zeros((self.height,self.width), dtype= np.float32)
        ave_arr_b = np.zeros((self.height,self.width), dtype= np.float32)
        ave_arr_r = np.zeros((self.height,self.width), dtype= np.float32)
        dframe_g = 0
        dframe_b = 0
        dframe_r = 0
        nframes = 10
        
        if  self.g_exists == 1:
            end = min(self.image_g.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            for j in range(anchor,  anchor+nframes):
                ave_arr_g = ave_arr_g + self.image_g[j]
            frame_g = ave_arr_g/(nframes)
            frame_g = rescale_intensity(frame_g, in_range = (minf,maxf), out_range = np.ubyte)
            dframe_g = bm3d.bm3d(frame_g, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)


        if  self.b_exists == 1:
            end = min(self.image_b.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            for j in range(start, end):
                ave_arr_b = ave_arr_b + self.image_b[j]
            frame_b = ave_arr_b/(nframes)
            frame_b = rescale_intensity(frame_b,in_range = (minf,maxf), out_range=np.ubyte)
            dframe_b = bm3d.bm3d(frame_b, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)


        if  self.r_exists == 1:
            end = min(self.image_r.shape[0], anchor+nframes)
            start = max(0, end - nframes)
            for j in range(start, end):
                ave_arr_r = ave_arr_r + self.image_r[j]
            frame_r = ave_arr_r/(nframes)
            frame_r = rescale_intensity(frame_r,in_range = (minf,maxf), out_range=np.ubyte)
            dframe_r = bm3d.bm3d(frame_r, 6, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)


        ch_dict = {'green' :  dframe_g,
                    'blue' :  dframe_b,
                    'red' :  dframe_r
                    }


        dframe = ch_dict[channel] 


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

        if np.any(dframe_b):
            self.dframe_b = dframe_b
            self.thres_b = np.median(dframe_b)*81
        else:
            self.dframe_b = dframe
            self.thres_b = np.median(dframe)*81

        if np.any(dframe_g):
            self.dframe_g = dframe_g
            self.thres_g = np.median(dframe_g)*81
        else:
            self.dframe_g = dframe
            self.thres_g = np.median(dframe)*81
        
        if np.any(dframe_r):
            self.dframe_r = dframe_r
            self.thres_r = np.median(dframe_r)*81
        else:
            self.dframe_r = dframe
            self.thres_r = np.median(dframe)*81
    
    
    
    
    def det_blob(self, plot = False, fsc = None, thres = None, r = 3, redchi_thres = 400):
        if thres != None:
            self.thres = thres
        print('Finding blobs')      
        blobs_dog = blob_dog(self.dcombined_image, min_sigma= (r-1) /sqrt(2), max_sigma = r /sqrt(2), threshold=self.thres, overlap=0, exclude_border = 2)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_dog=blobs_dog[blobs_dog[:,1].squeeze()<160]
        blobs_dog=blobs_dog[5<blobs_dog[:,1].squeeze()]
        if plot == True:
            fig = plt.figure()   
            ax = fig.add_subplot()
            ax.imshow(self.dcombined_image,cmap='Greys_r')
            
            for blob in blobs_dog:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='white',linewidth=0.5, fill=False)
                ax.add_patch(c)
            ax.set_axis_off()

            plt.tight_layout()
            
            
            self.cpath=os.path.join(self.path,r'circled')
            os.makedirs(self.cpath, exist_ok = True)
            plt.savefig(self.cpath+f'\\circle_{self.n_pro}.tif',dpi=300)
            plt.close()
        


        params = lmfit.Parameters()
        params.add('centery', value = 0)
        params.add('centerx', value = 0)
        params.add('amplitude', value = 5000)
        params.add('sigmay', value = 3)
        params.add('sigmax', value = 3)
        print(f'Found {blobs_dog.shape[0]} preliminary blobs')
       
        sigma_result = []
        coord_list = []
        quality = np.zeros(blobs_dog.shape[0])

        b_num = 0
       


        for blob in tqdm(np.arange(0,blobs_dog.shape[0])):

            try:
                fsc.set("progress", str(blob / (blobs_dog.shape[0]-1)))
            except:
                pass
            quality = 1
            y, x, r = blobs_dog[blob]
            r = 4
            y = round(y)
            x = round(x)

            redchi = 0
            redchi2 = 0
            redchi3 = 0
            rs1 = 0
            rs2 =0
            rs3 = 0
            nfev1 = 0
            nfev2 = 0
            nfev3 = 0 
            sigmay1 = 0
            sigmax1 = 0
            sigmay2 = 0
            sigmax2 = 0
            sigmay3 = 0
            sigmax3 = 0

            
            # check in range (overlayed)
            if (x-r)>10 and (x+r)<160 and(y-r)>1 and (y+r)<512:

                yf, xf = self.affine(x, y, self.M)
                ym = 0
                xm = 0

                yf = round(yf) 
                xf = round(xf) 
                ymf = 0
                xmf = 0
                
                
                yb, xb = self.affine(x, y, self.Mb)

                yb = round(yb) 
                xb = round(xb) + 342
                ymb = 0
                xmb = 0
                    

                y = round(y)
                x = round(x) + 171
                
                # gaussian fit green
                if (x-r) > 10 and (x+r) < 331 and(y-r) > 1 and (y+r) < 512 :
                    
                    z = self.dframe_g[y-r:y+r+1,x-r:x+r+1].flatten()
                    sum1 = np.sum(z)
                    if sum1 > self.thres_g: 
                        yr = np.zeros(81)
                        xr = np.zeros(81)
                        for j in range(0,81):
                            yr[j] = j//9
                            xr[j] = j%9
                        
                        model = lmfit.models.Gaussian2dModel()

                        result = model.fit(z, y=yr, x=xr,params=params, max_nfev = 150)
                        redchi = result.redchi
                        rs1 = result.rsquared
                        nfev1 = result.nfev

                        sigmay1 = result.best_values['sigmay']
                        sigmax1 = result.best_values['sigmax']
                        sigma_result.append((sigmay1, sigmax1, redchi))

                        if  0< result.best_values['centery'] <6 and 0 < result.best_values['centerx']<6 and redchi > 1:    
                            y = y + result.best_values['centery'] -4
                            x = x + result.best_values['centerx'] -4
                            ym = y - round(y)
                            xm = x - round(x)
                           
                        else: 
                            pass
                            #quality = 0
                        
                        redchi = result.redchi
                r = 4
                # gaussian fit red
                if  (xf-r)>10 and (xf+r)<160 and(yf-r)>1 and (yf+r)<511  :
                    
                    z = self.dframe_r[yf-r:yf+r+1,xf-r:xf+r+1].flatten()
                    sum2 = np.sum(z)
                    if sum2 > self.thres_r: 
                        yr=np.zeros((2*r+1)**2)
                        xr=np.zeros((2*r+1)**2)
                        for j in range(0,(2*r+1)**2):
                            yr[j]=j//(2*r+1)
                            xr[j]=j%(2*r+1)
                        model = lmfit.models.Gaussian2dModel()
                        result2 = model.fit(z, y=yr, x=xr,params=params, max_nfev = 150)
                        redchi2 = result2.redchi
                        rs2 = result2.rsquared
                        nfev2 = result2.nfev


                        sigmay2 = result2.best_values['sigmay']
                        sigmax2 = result2.best_values['sigmax']
                        sigma_result.append((sigmay2, sigmax2, redchi2))
                        
                        #print(redchi2)
                        if  0<result2.best_values['centery'] <6 and 0 < result2.best_values['centerx']<6 and redchi2 > 1:
                            yf = yf + result2.best_values['centery'] -r
                            xf = xf + result2.best_values['centerx'] -r
                            ymf = yf - round(yf)
                            xmf = xf - round(xf)
                        else:
                            pass
                            #quality =0

                # gaussian fit blue
                if  (xb-r)>352 and (xb+r)<500 and(yb-r)>1 and (yb+r)<511  and self.b_exists == 1:
                
                    z = self.dframe_b[yb-r:yb+r+1,xb-r:xb+r+1].flatten()
                    sum3 = np.sum(z)
                    if sum3 > self.thres_b: 
                        yr=np.zeros((2*r+1)**2)
                        xr=np.zeros((2*r+1)**2)
                        #print((2*r+1)**2)
                        for j in range(0,(2*r+1)**2):
                            yr[j] = j//(2*r+1)
                            xr[j] = j%(2*r+1)
                        model = lmfit.models.Gaussian2dModel()
                        result3 = model.fit(z, y=yr, x=xr,params=params, max_nfev = 150)
                        redchi3 = result3.redchi
                        rs3 = result3.rsquared
                        nfev3 = result3.nfev
                         
                        sigmay3 = result3.best_values['sigmay']
                        sigmax3 = result3.best_values['sigmax']
                        sigma_result.append((sigmay3, sigmax3, redchi3))
                        #print(redchi2)
                        if  0<result3.best_values['centery'] <6 and 0 < result3.best_values['centerx']<6 and redchi3 > 1:
                            yb = yb + result3.best_values['centery'] -r
                            xb = xb + result3.best_values['centerx'] -r
                            ymb = yb - round(yb)
                            xmb = xb - round(xb)
                        else:
                            pass
                            #quality =0
               
                r = 4
                y = round(y)
                x = round(x)
                yf = round(yf)
                xf = round(xf)  
                yb = round(yb)
                xb = round(xb)  
                    
                # max fit   
                if (x-r)>166 and (x+r)<336 and(y-r)>5 and (y+r)<507 and (xf-r)>5 and (xf+r)<166 and(yf-r)>5 and (yf+r)<507 and (xb-r)>337 and (xb+r)<507 and(yb-r)>5 and (yb+r)<507:

                    
                    #if redchi > 100  and ((result2.best_values['sigmax'] <7 and  result2.best_values['sigmay'] <7 ) or result2.redchi <3) :
                        max_location = np.argmax(self.dcombined_image[y-r:y+r+1,x-171-r:x-171+r+1])
                        ymax = max_location // 9 - 4
                        xmax = max_location % 9 - 4
                        c1 = ((redchi <  redchi_thres or (sigmay1 < 3.6 and sigmax1 < 3.6)) and redchi <  redchi_thres * 2)
                        c2 = ((redchi2 < redchi_thres or (sigmay2 < 3.6 and sigmax2 < 3.6)) and redchi2 <  redchi_thres * 2)
                        c3 = ((redchi3 <  redchi_thres or (sigmay3 < 3.6 and sigmax3 < 3.6) )and redchi3 <  redchi_thres * 2)
                        
                        if -2 < xmax <2 and -2< ymax < 2 and c1 and c2 and c3:
                            
                            if plot == True:
                                fig,axes=plt.subplots(1,3)
                                plt.axis('off')
                                axes[0].set_xticks([])
                                axes[1].set_xticks([])
                                axes[0].set_yticks([])
                                axes[1].set_yticks([])
                                axes[0].imshow(self.dframe_g[y-r:y+r+1,x-r:x+r+1],cmap='Greys_r',vmin=0,vmax=128)        
                                axes[1].imshow(self.dframe_g[yf-r:yf+r+1,xf-r:xf+r+1],cmap='Greys_r',vmin=0,vmax=128)
                                axes[2].imshow(self.dframe_b[yb-r:yb+r+1,xb-r:xb+r+1],cmap='Greys_r',vmin=0,vmax=160)  
                                axes[0].set_title(f'{sigmay1:.2f}, {sigmax1:.2f}, {redchi:.0f}')
                                axes[1].set_title(f'{sigmay2:.2f}, {sigmax2:.2f}, {redchi2:.0f}')
                                axes[2].set_title(f'{sigmay3:.2f}, {sigmax3:.2f}, {redchi3:.0f}')
                                plt.savefig(self.cpath+f'\\{b_num}.tif',dpi=300)
                                plt.close()

                            b_num = b_num + 1
                        else:
                            quality = 0                      
                 
                        if quality != 0:

                            coord_list.append((y, x, yf, xf, yb, xb, ym, xm, ymf, xmf, ymb, xmb))


        
        print(f'Found {len(coord_list)} filterd blobs')
        np.save(self.path+f'\\sigmas{self.n_pro}', np.array(sigma_result))
        return  coord_list
    
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

            y, x, yf, xf, yb, xb, ym, xm, ymf, xmf, ymb, xmb = blob
            r = 3

            
            y = int(y)
            x = int(x)
            yf = int(yf)
            xf = int(xf)
            yb = int(yb)
            xb = int(xb)


            if self.g_exists ==1:
                for t in range (0,self.g_length):
                    trace_gg[i][t] = np.sum(2 * self.gaussian_peaks(ym, xm) *(self.image_g[t][y-r:y+r+1,x-r:x+r+1]-self.bac_g[y-r:y+r+1,x-r:x+r+1]))
                    trace_gr[i][t] = np.sum(2 * self.gaussian_peaks(ymf, xmf)*(self.image_g[t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac_g[yf-r:yf+r+1,xf-r:xf+r+1]))
                    g_snap[blob_count][0] = self.image_g[:, y-4:y+4+1,x-4:x+4+1]
                    g_snap[blob_count][1] = self.image_g[:, yf-4:yf+4+1,xf-4:xf+4+1]
            
            if self.r_exists ==1:
                for t in range (0,self.r_length):           
                    trace_rr[i][t] = np.sum(2 * self.gaussian_peaks(ymf, xmf)*(self.image_r[t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac_r[yf-r:yf+r+1,xf-r:xf+r+1]))
                    r_snap[blob_count][0] = self.image_r[:,yf-4:yf+4+1,xf-4:xf+4+1]
                
            if  self.b_exists == 1:
                for t in range (0,self.b_length):     
                    trace_bb[i][t] = np.sum(2 * self.gaussian_peaks(ymb, xmb)*(self.image_b[t][yb-r:yb+r+1,xb-r:xb+r+1]-self.bac_b[yb-r:yb+r+1,xb-r:xb+r+1]))
                    trace_bg[i][t] = np.sum(2 * self.gaussian_peaks(ym, xm)*(self.image_b[t][y-r:y+r+1,x-r:x+r+1]-self.bac_b[y-r:y+r+1,x-r:x+r+1]))
                    trace_br[i][t] = np.sum(2 * self.gaussian_peaks(ymf, xmf)*(self.image_b[t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac_b[yf-r:yf+r+1,xf-r:xf+r+1]))
                    b_snap[blob_count][0] = self.image_b[:,yb-4:yb+4+1,xb-4:xb+4+1]
                    b_snap[blob_count][1] = self.image_b[:,y-4:y+4+1,x-4:x+4+1]
                    b_snap[blob_count][2] = self.image_b[:,yf-4:yf+4+1,xf-4:xf+4+1]

            i=i+1
             
        trace_gg = trace_gg[0:i]
        trace_gr = trace_gr[0:i]
        trace_rr = trace_rr[0:i]
        trace_bb = trace_bb[0:i]
        trace_bg = trace_bg[0:i]
        trace_br = trace_br[0:i]
        
        np.savez(self.path + r'\blobs.npz', b = b_snap, g = g_snap, r = r_snap, minf = minf, maxf = maxf)
        
        return trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i
        
        
