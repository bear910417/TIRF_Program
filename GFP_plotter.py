import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import os



class GFP:
    def __init__(self,path,select):
        self.path = path
        self.select = select
        if self.select !=-1:
            try:
                self.selected = np.load(self.path+r'\\selected.npy')
            except:
                self.selected = np.ones(500)
                    
        else:
            self.selected = np.ones(500)

    def plot(self, plot):
        
        file_path = self.path + r'\\data.npz'
        
        
        #plot = False
        
        plt.close()
        avg_b= np.load(file_path)['avg_b']
        avg_donor=np.load(file_path)['avg_donor']
        avg_acceptor=np.load(file_path)['avg_acceptor']
        fret = np.load(file_path)['fret'] 
        time_gr = np.load(file_path)['time_gr']
        time_b = np.load(file_path)['time_b']
        
        os.makedirs(self.path +r'\\gfp_scatter', exist_ok = True)
        
        if plot == True:
            os.makedirs(self.path +r'\\gfp', exist_ok = True)
            
            for i in tqdm(np.arange(0,avg_b.shape[0])):
                plt.rcParams["figure.figsize"] = (15,6)
                fig, axs = plt.subplots(2)
                axs[0].plot(time_b, avg_b[i])
                axs[0].set(ylim=(0,5000),ylabel='Intensity')
                
                axs[1].plot(time_gr,fret[i])
                axs[1].set(ylim=(0,1.0),xlabel='time(frame)',ylabel='FRET',yticks=np.arange(0.0,1.0,0.1)) 
                axs[1].grid(True,axis='y')
                
                plt.savefig(self.path +f'\\gfp\\trace{i}.tif', dpi = 200)
                plt.close()
                plt.rcParams["figure.figsize"] = (10,10)
                plt.scatter(fret[i], avg_b[i], s = 1)
                plt.xlim(0,1)
                plt.ylim(0,3000)
                plt.xlabel('FRET')
                plt.ylabel('GFP Intensity')
                plt.savefig(self.path +f'\\gfp_scatter\\trace_{i}.tif', dpi = 200)
                plt.close()
                
        font = {'family': 'Arial',
                'size': 5,
                }

        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['ytick.major.width'] = 0.5
        plt.rc('font', **font)
        plt.rc('xtick', labelsize=5) 
        plt.rc('ytick', labelsize=5) 
        px = 1/72
        plt.rcParams["figure.figsize"] = (100*px,105*px)
        fret_selected = []
        avg_b_selected=[]
        for i in range (fret.shape[0]):
            if self.selected[i]!=-1:
                fret_selected.append(fret[i])
                avg_b_selected.append(avg_b[i])
        plt.scatter(fret_selected, avg_b_selected, marker = '.', s = 0.5, edgecolors='None')
        plt.hlines(400,0,1,colors='skyblue', linestyles='dashed')
        plt.xlim(0,1)
        plt.ylim(0,1500)
        plt.xlabel('FRET')
        plt.ylabel('GFP Intensity')
        plt.tight_layout()
        plt.savefig(self.path +r'\\gfp_scatter\\all.eps', dpi = 1200, format = 'eps')
        #plt.show()
        plt.close()
        
        heatmap, xedges, yedges = np.histogram2d(np.array(fret_selected).reshape(-1),  np.array(avg_b_selected).reshape(-1), bins=50, range =  [[0, 1], [0, 1000]], density = True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, ax = plt.subplots(figsize=(120/72,100/72), ncols=1)
        plt.xticks(np.arange(0,1.1,0.2))
        plt.yticks(np.arange(0,1001,500))
        plt.xlabel('FRET')
        plt.ylabel('GFP Intensity')
        pos = plt.imshow(heatmap.T,  extent = extent, aspect = 'auto', origin='lower', cmap='Greys')
        pos.set_clim(0, 0.03)
        cbar = fig.colorbar(pos, ax=ax, format = "%4.2f", ticks = [0, 0.01, 0.02, 0.03])
        plt.hlines(400,0,1,colors='skyblue', linestyles='dashed')
        plt.tight_layout()
        plt.savefig(self.path +r'\\gfp_scatter\\all_heat.eps', dpi = 1200, format = 'eps')
        plt.show()