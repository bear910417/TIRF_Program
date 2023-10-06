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
        self.frame_b = None
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
        
        
        self.gaussian_peaks = np.zeros((7,7,7,7),dtype=np.float32)
        for k in range (0,7):
            for l in range (0,7):     
              offy = -0.5*float(k)
              offx = -0.5*float(l)
              
              for i in range (0,7): 
                for j in range (0,7):
                  dist = 0.3 * ((float(i)+offy-1.5)**2 + (float(j)+offx-1.5)**2)
                  self.gaussian_peaks[k][l][i][j]= np.exp(-dist)
        
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
                    gfilename = str(1) + '.glimpse'
                    gfile_path = path_g+r'\\'+gfilename
                    image_g_1 = np.fromfile(gfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
                    image_g = np.concatenate((image_g,image_g_1))
                except:
                    pass
    
            image_g = image_g + 2**16
        
        
        #r_exist?
        time_r = np.zeros(10)
        image_r = np.zeros((1, 512, 512))
        if  r_exists == 1:
            rfilename = str(filenumber[0]) + '.glimpse'
            rfile_path = path_r+r'\\'+rfilename
            time_r = self.cal_time(path_r, self.r_start, self.r_length, first)
            image_r = np.fromfile(rfile_path, dtype=(np.dtype('>i2') , (self.height,self.width)))
            image_r = image_r + 2**16
        
            
        #b_exist?    
        time_b = np.zeros(10)
        image_b = np.zeros((10, 512, 512))

        #calculate backgrounds
        bac_mode = self.bac_mode 
         
        if  b_exists == 1:
            print(f'Calculating b Backgrounds with mode {bac_mode}')  
            file = h5py.File(path_b+r'\header.mat','r')
            bfilename = str(filenumber[0]) + '.glimpse'
            bfile_path = path_b+r'\\'+bfilename

            if first == None:
                nframes_true = int(file[r'/vid/nframes'][0][0])
                time_b, first = self.cal_time_g(path_b, self.b_start, self.b_length)
            else:
                time_b = self.cal_time(path_b, self.b_start, self.b_length, first)
            image_b = np.fromfile(bfile_path, dtype=(np.dtype('>i2') , (self.height, self.width)))
            image_b = image_b + 2**16
            
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
                        #aves[int((i-8)/16)][int((j-8)/16)] = np.round(np.median(bac_temp[i-8:i+8,j-8:j+8]),1)
                        aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw],0.5),1)
                        #print(i/bw,j/bw)
                #print(aves.shape)
                aves =  scipy.ndimage.zoom(aves,bw,order=1)
                bac_b.append(aves)   
                
            self.bac_b=np.average(bac_b,axis=0)

            try:
                fsc.set("load_progress", '0')
            except:
                pass

        ###

        print(f'Calculating g Backgrounds with mode {bac_mode}')  
        bac=[]
        for bt in range(0,nframes):
            try:
                fsc.set("load_progress", str(bt / (nframes-1) - 0.4))
            except:
                pass
            
            bac_temp = image_g[bt]
            if bac_mode == 0:
                bac_temp = scipy.ndimage.filters.uniform_filter(bac_temp,size=3,mode='nearest')
            
            bw = 16 
            aves = np.zeros((int(self.height / bw),int(self.height / bw)), dtype= np.float32)
            
            for i in range(0,self.height, bw): #0~480
                for j in range(0,self.width, bw): #0~480
                    if bac_mode == 0:
                        aves[int((i-8)/16)][int((j-8)/16)] = np.round(np.amin(bac_temp[i:i+bw,j:j+bw]),1)
                    else:
                        aves[int(i/bw)][int(j/bw)] = np.round(np.quantile(bac_temp[i:i+bw,j:j+bw],0.4),1)
                    #print(i/bw,j/bw)
            #print(aves.shape)
            aves =  scipy.ndimage.zoom(aves,bw,order=1)
            #print(aves.shape)
            if bac_mode ==0:
                aves = scipy.ndimage.filters.uniform_filter(aves,size=21,mode='nearest')
            bac.append(aves)
        self.bac = np.average(bac,axis=0)


        #rescale image intensity and remove background

        self.image_g = image_g
        self.image_r = image_r
        self.image_b = image_b
        self.b_exists = b_exists
        
        return  time_g, time_r, time_b, nframes_true
    
    
    
    def gen_dimg(self, anchor, mpath, maxf = 35000, minf = 32946, channel = 'green', plot = True):
        
        path = self.path 
        ch_dict = {'green' :  self.image_g,
         'blue' :  self.image_b
        }

        st_dict = {'green' :  self.g_start,
         'blue' :  self.b_start
        }

        if np.any(ch_dict['blue']):
            target_image = (ch_dict['green'] + ch_dict['blue'] ) / 2
        else:
            target_image = ch_dict[channel] 

        start = st_dict[channel]

        frame = np.zeros((self.height,self.width), dtype= np.int16)
        ave_arr = np.zeros((self.height,self.width), dtype= np.float32)
        ave_arr_b = np.zeros((self.height,self.width), dtype= np.float32)
        nframes = 10
        
        for j in range(start+anchor,  start+anchor+nframes):
            ave_arr= ave_arr + target_image[j]
            
        ave_arr = ave_arr/(nframes)
        frame = ave_arr
        
        
        
        temp1 = frame 
        temp1 =scipy.ndimage.filters.uniform_filter(temp1,size=3,mode='nearest')
        
        frame = rescale_intensity(frame,in_range=(minf,maxf),out_range=np.ubyte)
        dframe = bm3d.bm3d(frame, 6, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        
        frame_b = 0
        if  self.b_exists == 1:
            for j in range(self.b_start+anchor, self.b_start+anchor+nframes):
                ave_arr_b= ave_arr_b + self.image_b[j]
            ave_arr_b = ave_arr_b/(nframes)
            frame_b = ave_arr_b
            temp1_b = frame_b
            maxf=np.max(frame_b)
            minf=np.min(frame_b)
            temp1_b =scipy.ndimage.filters.uniform_filter(temp1_b,size=3,mode='nearest')
            frame_b = rescale_intensity(frame_b,in_range=(minf,maxf),out_range=np.ubyte)
            
        if plot == True:
            plt.imshow(np.concatenate((frame,dframe),axis=1),cmap='Greys_r',vmin=0,vmax=128)
            plt.savefig(self.path+r'\\ave.tif',dpi=300)
            plt.close()
        temp1=dframe


        #combine two channel image
        self.M = np.load(mpath + r'\map_g_r.npy')
        self.Mb = np.load(mpath + r'\map_g_b.npy')



        left_image  = temp1[0:self.height,0:170]
        right_image = temp1[0:self.height,171:341]
        blue_image = temp1[0:self.height,342:512]
        rows, cols = right_image.shape


        
        left_image_trans=cv2.warpAffine(left_image, self.M, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
        blue_image_trans=cv2.warpAffine(blue_image, self.Mb, (cols, rows), flags = cv2.WARP_INVERSE_MAP)
             
        dcombined_image = (right_image + left_image_trans + blue_image_trans)
        
        if plot == True:
            plt.imshow(dcombined_image ,cmap='Greys_r')
            plt.savefig(path+f'\\combined_image_{self.n_pro}.tif',dpi=300)
            plt.close()
        toc = time.perf_counter()
        print(f"Finished in {toc - self.tic:0.4f} seconds")
        
        
        self.dcombined_image = dcombined_image
        
        self.dframe = dframe
        self.frame_b = frame_b
    
    
    
    
    def det_blob(self, plot = True, fsc = None, thres = None, r = 3):
        if thres != None:
            self.thres = thres
        print('Finding blobs')      
        blobs_dog = blob_dog(self.dcombined_image, min_sigma= (r-1) /sqrt(2), max_sigma = r /sqrt(2), threshold=self.thres, overlap=0, exclude_border=2)
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
            if not os.path.exists(self.cpath):
                os.makedirs(self.cpath)
                
            plt.savefig(self.cpath+f'\\circle_{self.n_pro}.tif',dpi=300)
            plt.show
            plt.close()
        
        gaussian_peaks2 = self.gaussian_peaks


        params = lmfit.Parameters()
        params.add('centery',value = 0)
        params.add('centerx',value = 0)
        params.add('amplitude',value = 5000)
        params.add('sigmay',value = 3)
        params.add('sigmax',value = 3)
        print(f'Found {blobs_dog.shape[0]} preliminary blobs')
       

        img_g_ave = np.average(self.image_g[self.g_start:self.g_start+10],axis=0)


        if  self.b_exists == 1:
            img_b_ave = np.average(self.image_b[self.b_start:self.b_start+10],axis=0)
            
        sigma_result = []
        coord_list = []
        quality = np.zeros(blobs_dog.shape[0])

        for blob in tqdm(np.arange(0,blobs_dog.shape[0])):

            try:
                fsc.set("progress", str(blob / (blobs_dog.shape[0]-1)))
            except:
                pass
            quality = 1
            y, x, r = blobs_dog[blob]
            r = 4
            y=round(y)
            x=round(x)
            # check in range (overlayed)
            if (x-r)>10 and (x+r)<160 and(y-r)>1 and (y+r)<512:

                yf, xf = self.affine(x, y, self.M)

                yf = round(yf) 
                xf = round(xf) 
                
                
                yb, xb = self.affine(x, y, self.Mb)

                yb = round(yb) 
                xb = round(xb) + 342
                ymb = 0
                xmb = 0
                    

                y=round(y)
                x=round(x) 
                
                # gaussian fit green
                if (x-r)>10 and (x+r)<160 and(y-r)>1 and (y+r)<512 :
                    
                    z=self.dcombined_image[y-r:y+r+1,x-r:x+r+1].flatten()
                    if np.sum(z) > 500: 
                        yr=np.zeros(81)
                        xr=np.zeros(81)
                        for j in range(0,81):
                            yr[j]=j//9
                            xr[j]=j%9
                        
                        model = lmfit.models.Gaussian2dModel()

                        result = model.fit(z, y=yr, x=xr,params=params)
                        redchi = result.redchi
                        if  0<result.best_values['centery'] <6 and 0 < result.best_values['centerx']<6 and redchi>100:       
                            y = y + result.best_values['centery'] -4
                            x = x + result.best_values['centerx'] -4
                            
                        else: 
                            pass
                            #quality = 0
                        
                        redchi = result.redchi
                
                    
                
                y=round(y)
                x=round(x) + 171
                
                r =4
                # gaussian fit red
                if  (xf-r)>10 and (xf+r)<160 and(yf-r)>1 and (yf+r)<511  :
                    
                    z=self.dframe[yf-r:yf+r+1,xf-r:xf+r+1].flatten()
                    if np.sum(z) > 500: 
                        yr=np.zeros((2*r+1)**2)
                        xr=np.zeros((2*r+1)**2)
                        for j in range(0,(2*r+1)**2):
                            yr[j]=j//(2*r+1)
                            xr[j]=j%(2*r+1)
                        model = lmfit.models.Gaussian2dModel()
                        result2 = model.fit(z, y=yr, x=xr,params=params)
                        redchi2 = result2.redchi
                        sigmay = result2.best_values['sigmay']
                        sigmax = result2.best_values['sigmax']
                        sigma_result.append((sigmay, sigmax, redchi2))
                        
                        #print(redchi2)
                        if  0<result2.best_values['centery'] <6 and 0 < result2.best_values['centerx']<6 and redchi2>100:
                            yf = yf + result2.best_values['centery'] -r
                            xf = xf + result2.best_values['centerx'] -r
                        else:
                            pass
                            #quality =0

                # gaussian fit blue
                if  (xb-r)>352 and (xb+r)<500 and(yb-r)>1 and (yb+r)<511  and self.b_exists == 1:
                
                    z=self.dframe[yb-r:yb+r+1,xb-r:xb+r+1].flatten()
                    if np.sum(z) > 500: 
                        yr=np.zeros((2*r+1)**2)
                        xr=np.zeros((2*r+1)**2)
                        #print((2*r+1)**2)
                        for j in range(0,(2*r+1)**2):
                            yr[j]=j//(2*r+1)
                            xr[j]=j%(2*r+1)
                        model = lmfit.models.Gaussian2dModel()
                        result3 = model.fit(z, y=yr, x=xr,params=params)
                        redchi3 = result3.redchi
                        sigmay = result3.best_values['sigmay']
                        sigmax = result3.best_values['sigmax']
                        sigma_result.append((sigmay, sigmax, redchi3))
                        
                        #print(redchi2)
                        if  0<result3.best_values['centery'] <6 and 0 < result3.best_values['centerx']<6 and redchi3>100:
                            yb = yb + result3.best_values['centery'] -r
                            xb = xb + result3.best_values['centerx'] -r
                        else:
                            pass
                            #quality =0
               
                r = 4
                yf=round(yf)
                xf=round(xf)  
                yb=round(yb)
                xb=round(xb)  
                    
                # max fit   
                if (x-r)>166 and (x+r)<336 and(y-r)>5 and (y+r)<507 and (xf-r)>5 and (xf+r)<166 and(yf-r)>5 and (yf+r)<507 and (xb-r)>337 and (xb+r)<507 and(yb-r)>5 and (yb+r)<507:

                    
                    # if redchi < 150  and ((result2.best_values['sigmax'] <7 and  result2.best_values['sigmay'] <7 ) or result2.redchi <3) :
                        max_location = np.argmax(self.dcombined_image[y-r:y+r+1,x-171-r:x-171+r+1])

                        ym = max_location // 9 - 4
                        xm = max_location % 9 - 4
                        
                        if -2 < xm <2 and -2< ym < 2 :

                            pass
                        else:
                            
                            xm=0
                            ym=0
                            quality = 0                      
                        
                        r=3
                        max_value = np.max(img_g_ave[y-r:y+r+1,x-r:x+r+1]-self.bac[y-r:y+r+1,x-r:x+r+1])       
                       
                        
                        diff=np.zeros((7,7))
                        for xm in range(0,7):
                            for ym in range(0,7):
                                diff[ym,xm] = np.abs(np.sum(max_value*gaussian_peaks2[ym][xm]-(img_g_ave[y-r:y+r+1,x-r:x+r+1]-self.bac[y-r:y+r+1,x-r:x+r+1])))
                        am = np.argmin(diff)  
                        ym = am // 7 
                        xm = am % 7 

                
                        max_value = np.max(img_g_ave[yf-r:yf+r+1,xf-r:xf+r+1]-self.bac[yf-r:yf+r+1,xf-r:xf+r+1])
                        diff=np.zeros((7,7))
                        for xmf in range(0,7):
                            for ymf in range(0,7):
                                diff[ymf,xmf] = np.abs(np.sum(max_value*gaussian_peaks2[ymf][xmf]-(img_g_ave[yf-r:yf+r+1,xf-r:xf+r+1]-self.bac[yf-r:yf+r+1,xf-r:xf+r+1])))
                        amf = np.argmin(diff)   
                        ymf = amf // 7
                        xmf = amf % 7
                            
                        if self.b_exists ==1:
                            max_value = np.max(img_b_ave[yb-r:yb+r+1,xb-r:xb+r+1]-self.bac_b[yb-r:yb+r+1,xb-r:xb+r+1])
                            diff=np.zeros((7,7))
                            for xmb in range(0,7):
                                for ymb in range(0,7):
                                    diff[ymb,xmb] = np.abs(np.sum(max_value*gaussian_peaks2[ymb][xmb]-(img_b_ave[yb-r:yb+r+1,xb-r:xb+r+1]-self.bac[yb-r:yb+r+1,xb-r:xb+r+1])))
                            amb = np.argmin(diff)   
                            ymb = amb // 7
                            xmb = amb % 7
                        if quality != 0:
                            coord_list.append((y, x, yf, xf, yb, xb, ym, xm, ymf, xmf, ymb, xmb))
        
        print(f'Found {len(coord_list)} filterd blobs')
        np.save(self.path+f'\\sigmas{self.n_pro}', np.array(sigma_result))
        return  coord_list
    
    def cal_intensity(self, coord_list, drifts, space, cal_drift, plot, fsc = None):
        
        print('Calcultating Intensities')
        i=0

        trace_gg = np.zeros((1000,int(self.g_length)))
        trace_gr = np.zeros((1000,int(self.g_length)))
        trace_rr = np.zeros((1000,int(self.r_length)))
        trace_bb = np.zeros((1000,int(self.b_length)))
        trace_bg = np.zeros((1000,int(self.b_length)))
        trace_br = np.zeros((1000,int(self.b_length)))
        self.cpath=os.path.join(self.path,r'circled')
        os.makedirs(self.cpath+f'\\{self.n_pro}', exist_ok=True)
        for blob_count, blob in enumerate(coord_list):

            try:
                fsc.set("cal_progress", str(blob_count / (len(coord_list)-1)))
            except:
                pass

            y, x, yf, xf, yb, xb, ym, xm, ymf, xmf, ymb, xmb = blob
            r = 3

            if plot == True:
                fig,axes=plt.subplots(1,3)
                plt.axis('off')
                axes[0].set_xticks([])
                axes[1].set_xticks([])
                axes[0].set_yticks([])
                axes[1].set_yticks([])
                axes[0].imshow(self.dframe[y-r:y+r+1,x-r:x+r+1],cmap='Greys_r',vmin=0,vmax=128)        
                axes[1].imshow(self.dframe[yf-r:yf+r+1,xf-r:xf+r+1],cmap='Greys_r',vmin=0,vmax=128)
                if  self.b_exists == 1:
                    axes[2].imshow(self.frame_b[yb-r:yb+r+1,xb-r:xb+r+1],cmap='Greys_r',vmin=0,vmax=160)  
                else:
                    axes[2].imshow(self.dcombined_image[y-r:y+r+1,x-171-r:x-171+r+1],cmap='Greys_r',vmin=0,vmax=160)  
                plt.savefig(self.cpath+f'\\{self.n_pro}\\'+str(i//2)+'.tif',dpi=300)
                plt.close()
                

            y_old = y
            x_old = x
            yf_old = yf
            xf_old = xf
           
            y_start = y_old 
            x_start = x_old 
            yf_start = yf_old 
            xf_start = xf_old 
            for t in range (0,self.g_length):
                
                if cal_drift ==1:
                    j = int(t / space)
                    #print(j)
                    drift = drifts[j]
    
    
                    y = (y_start +drift[0])
                    x = (x_start + drift[1]) 
                    yf = (yf_start +drift[2])
                    xf = (xf_start +drift[3])
                
                    y_start = y
                    x_start = x
                    yf_start = yf
                    xf_start = xf
                    
                    y = round(y)
                    x = round(x)
                    yf = round(yf)
                    xf = round(xf)
                #print(t,y,x,yf,xf)
                trace_gg[i][t]=np.sum(2*self.gaussian_peaks[ym][xm]*(self.image_g[self.g_start+t][y-r:y+r+1,x-r:x+r+1]-self.bac[y-r:y+r+1,x-r:x+r+1]))
                trace_gr[i][t]=np.sum(2*self.gaussian_peaks[ymf][xmf]*(self.image_g[self.g_start+t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac[yf-r:yf+r+1,xf-r:xf+r+1]))
            
            if self.r_exists ==1:
                for t in range (0,self.r_length):
                           
                    trace_rr[i][t]=np.sum(2*self.gaussian_peaks[ymf][xmf]*(self.image_r[self.r_start+t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac[yf-r:yf+r+1,xf-r:xf+r+1]))
                
            if  self.b_exists == 1:
                for t in range (0,self.b_length):
                   
                    trace_bb[i][t]=np.sum(2*self.gaussian_peaks[ymb][xmb]*(self.image_b[self.b_start+t][yb-r:yb+r+1,xb-r:xb+r+1]-self.bac_b[yb-r:yb+r+1,xb-r:xb+r+1]))
                    trace_bg[i][t]=np.sum(2*self.gaussian_peaks[ym][ym]*(self.image_b[self.b_start+t][y-r:y+r+1,x-r:x+r+1]-self.bac_b[y-r:y+r+1,x-r:x+r+1]))
                    trace_br[i][t]=np.sum(2*self.gaussian_peaks[ymf][ymf]*(self.image_b[self.b_start+t][yf-r:yf+r+1,xf-r:xf+r+1]-self.bac_b[yf-r:yf+r+1,xf-r:xf+r+1]))

            i=i+1
            #print(self.bac_b)
             
        trace_gg = trace_gg[0:i]
        trace_gr = trace_gr[0:i]
        trace_rr = trace_rr[0:i]
        trace_bb = trace_bb[0:i]
        trace_bg = trace_bg[0:i]
        trace_br = trace_br[0:i]
        
        #print(str(trace.shape[0]//2)+" blobs were found\n")
        
        return trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i
        
        
