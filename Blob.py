import numpy as np
import lmfit
import matplotlib.pyplot as plt


class Blob():
    
    def __init__(self, raw_blob, M, Mb):
        self.org_y = int(raw_blob[0])
        self.org_x = int(raw_blob[1])
        self.r = 4


        self.coords = np.zeros((3, 2))

        self.M = M
        self.Mb = Mb
      

        self.shift = np.zeros((3, 2))

        params = lmfit.Parameters()
        params.add('centery', value = 4, min = 3, max = 5)
        params.add('centerx', value = 4, min = 3, max = 5)
        params.add('amplitude', value = 5000)
        params.add('sigmay', value = 3, min = 0, max = 6)
        params.add('sigmax', value = 3, min = 0, max = 6)
        self.params = params


        self.sum = np.zeros(3)
        self.redchi = np.zeros(3)
        self.rs = np.zeros(3)
        self.nfev = np.zeros(3)
        self.sigma = np.zeros((3, 2))
        self.center = np.zeros((3, 2))

        self.quality = 1

    def affine(self, y, x, M, x_shift = 0):
        x1 = round(M[0][0] * x + M[0][1] * y + M[0][2]) + x_shift
        y1 = round(M[1][0] * x + M[1][1] * y + M[1][2])
        
        return [y1, x1]


    def map_coord(self):
        self.coords[0] = self.affine(self.org_y, self.org_x, self.M, x_shift = 0)
        self.coords[2] = self.affine(self.org_y, self.org_x, self.Mb, x_shift = 342)
        self.coords[1] = [round(self.org_y), round(self.org_x) + 171]

    def check_bound(self):

        r = self.r

        # check original boundary
        if (self.org_x - r) < 1 or (self.org_x + r) > 170 or (self.org_y - r) < 1 or (self.org_y + r) > 511:
            self.quality = 0

        # check red
        if (self.coords[0][1] - r) < 1 or (self.coords[0][1] + r) > 171 or (self.coords[0][0] - r) < 1 or (self.coords[0][0] + r) > 511:
            self.quality = 0
        
        # check green
        if (self.coords[1][1] - r) < 172 or (self.coords[1][1] + r) > 341 or (self.coords[1][0] - r) < 1 or (self.coords[1][0] + r) > 511:
            self.quality = 0

        #check red
        if (self.coords[2][1] - r) < 342 or (self.coords[2][1] + r) > 511 or (self.coords[2][0] - r) < 1 or (self.coords[2][0] + r) > 511:
            self.quality = 0

    def check_max(self, dcombined_image):
        if self.quality == 0:
            return None
        r = 4
        max_location = np.argmax(dcombined_image[self.org_y - r : self.org_y + r + 1, self.org_x  - r : self.org_x + r + 1])
        ymax = max_location // 9 - 4
        xmax = max_location % 9 - 4
        if not(-2 < xmax <2 and -2< ymax < 2):
            self.quality = 0
        
    def set_image(self, image, channel):
        if channel == 'red':
            self.dframe_r = image
        elif channel == 'green':
            self.dframe_g = image
        elif channel == 'blue':
            self.dframe_b = image
        

    def set_params(self, ch):
        self.params['centery'].set(value = 4 + self.shift[ch][0], min = 4 + self.shift[ch][0] - 0.5, max = 4 + self.shift[ch][0] + 0.5)
        self.params['centerx'].set(value = 4 + self.shift[ch][1], min = 4 + self.shift[ch][1] - 0.5, max = 4 + self.shift[ch][1] + 0.5)
        self.params['amplitude'].set(value = self.sum[ch], vary = False)
        self.params['sigmay'].set(value = self.sigma[ch][0], vary = False)
        self.params['sigmax'].set(value = self.sigma[ch][1], vary = False)

        
    def gaussian_fit(self, ch, nfev = 150):
        if self.quality == 0:
            return None
        
        ch_dict = {0 :  self.dframe_r,
                    1 :  self.dframe_g,
                    2 :  self.dframe_b,}
        
        image = ch_dict[ch]
        thres = np.median(image)*81
        
        r = 4
        y = int(np.round(self.coords[ch][0]))
        x = int(np.round(self.coords[ch][1]))
        z = image[y-r:y+r+1,x-r:x+r+1].flatten()
        sum = np.sum(z)

        if sum > thres: 
            yr = np.arange(0, 9)
            xr = np.arange(0, 9)
            yr, xr = np.meshgrid(yr, xr, indexing='ij')
            yr = yr.flatten()
            xr = xr.flatten()
            
            model = lmfit.models.Gaussian2dModel()

            result = model.fit(z, y = yr, x = xr, params = self.params, max_nfev = nfev)
            self.redchi[ch] = result.redchi
            self.rs[ch] = result.rsquared
            self.sum[ch] = result.best_values['amplitude']
            self.nfev[ch] = result.nfev
            self.sigma[ch] = [result.best_values['sigmay'], result.best_values['sigmax']]
            self.center[ch] = [result.best_values['centery'], result.best_values['centerx']]

            if self.redchi[ch] > 1:
                self.coords[ch] = [y + result.best_values['centery'] - 4, x + result.best_values['centerx'] - 4]
                self.shift[ch] = self.coords[ch] - np.round(self.coords[ch])
                self.coords[ch] = np.round(self.coords[ch])
                

    def check_fit(self, redchi_thres):
        for ch in range(0, 3):
            c1 = (self.redchi[ch] <  redchi_thres or (self.sigma[ch][0] < 3.6 and self.sigma[ch][1] < 3.6)) 
            c2 = self.redchi[ch] <  redchi_thres * 2
            if not(c1 and c2):
                self.quality = 0


    def plot_circle(self, dframe_g, dframe_b, i):
        r = 4
        yr = self.coords[0][0]
        xr = self.coords[0][1]
        yg = self.coords[1][0]
        xg = self.coords[1][1]
        yb = self.coords[2][0]
        xb = self.coords[2][1]

        fig, axes = plt.subplots(1,3)
        plt.axis('off')
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_yticks([])

        axes[0].imshow(dframe_g[yr-r:yr+r+1, xr-r:xr+r+1], cmap='Greys_r', vmin=0, vmax=128)        
        axes[1].imshow(dframe_g[yg-r:yg+r+1, xg-r:xg+r+1], cmap='Greys_r', vmin=0, vmax=128)
        axes[2].imshow(dframe_b[yb-r:yb+r+1, xb-r:xb+r+1], cmap='Greys_r', vmin=0, vmax=160)  
        plt.savefig(self.cpath+f'\\{i}.tif',dpi=300)
        plt.close()

    def get_coord(self):
        c = list(self.coords.flatten())
        m = list(self.shift.flatten())
        return c + m


        






        
        


       


            

        

    
        

    





    

