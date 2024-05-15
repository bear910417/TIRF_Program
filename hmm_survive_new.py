import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from surpyval import Exponential
import seaborn as sns
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from tqdm import tqdm

def process(path):
    hds = np.load(path + r'\\HMM_traces\\hmm.npz', allow_pickle = True)['hd_states']
    try:
         times = np.load(path + r'\\data.npz', allow_pickle = True)['time_g']
    except:
         times = np.arange(0, 1800-0.05, 0.05)
    means = np.load(path + r'\\HMM_traces\\hmm.npz', allow_pickle = True)['means'].reshape(-1)
    means = np.flip(np.sort(means))
    tot_states = means.shape[0]
    dwells = []
    dwell_1 = []
    d_10 = 0
    d_12 = 0
    for i, trace in enumerate(tqdm(hds)):
        time = times
        diff = np.ediff1d(trace)
        trans = np.nonzero(diff)[0]
        for j in range(0, trans.shape[0]-1):
                dwell = time[trans[j+1]] - time[trans[j]]
                org = np.argwhere(means == trace[trans[j]+1])[0][0]
                des = np.argwhere(means == trace[trans[j+1]+1])[0][0]
                dwells.append((dwell, org, des))
    #k on
    kon  = []
    koff  = []
    for org in tqdm(range(0, tot_states - 1)):
            des = org + 1
            #print(f'Processing {org} --> {des}.')
            dwell_times = []
            for d in dwells:
                if d[1] == org and d[2] == des:
                    dwell_times.append(d[0])
                if d[1] == 1 and d[2] == des:
                    dwell_1.append(d[0])
                    d_12 += 1

            if dwell_times == []:
                dwell_times = [1]

            num_events = len(dwell_times)
            #print(num_events)
            model = Exponential.fit(x = dwell_times)
            k = model.params[0]
            #print(k)
            d_time = 1 / k

            t = np.arange(0, 10, 0.1)
            kmf = KaplanMeierFitter()
            kmf.fit(dwell_times, timeline = t, alpha = 0.3174)
            ci = kmf.confidence_interval_
            s = kmf.survival_function_
            plt.plot(t, s['KM_estimate'], label = 't2')
            plt.fill_between(t, ci[r'KM_estimate_lower_0.6826'], ci[r'KM_estimate_upper_0.6826'], alpha = 0.2)
            plt.xlim(0, 10)
            plt.ylim(0,1)
            sns.despine()
            plt.text(5, 0.5,f' N = {num_events} \n kon = {k:0.4f} s$^{{{-1}}}$ \n \u03C4 = {d_time:0.4f} s',ha='left', fontsize='large')
            plt.title(f'{org} -> {des}')
            plt.xlabel('time (s)')
            plt.ylabel('Cumulative Probability')
            #plt.tight_layout()
            os.makedirs(path + r'\\k', exist_ok = True)
            plt.savefig(path + f'\\k\\{org}_{des}_off.tif', dpi = 1200, format = 'tif')
            plt.close()
            kon.append(k)

    #k off
    for org in tqdm(range(1, tot_states)):
            des = org - 1
            #print(f'Processing {org} --> {des}.')
            dwell_times = []
            for d in dwells:
                if d[1] == org and d[2] == des:
                    dwell_times.append(d[0])
                if d[1] == 1 and d[2] == des:
                    dwell_1.append(d[0])
                    d_10 += 1

            num_events = len(dwell_times)
            #print(num_events)
            if dwell_times == []:
                dwell_times = [1]
            model = Exponential.fit(x = dwell_times)
            k = model.params[0]
            #print(k)
            d_time = 1 / k

            t = np.arange(0, 10, 0.1)
            kmf = KaplanMeierFitter()
            kmf.fit(dwell_times, timeline = t, alpha = 0.3174)
            ci = kmf.confidence_interval_
            s = kmf.survival_function_
            plt.plot(t, s['KM_estimate'], label = 't2')
            plt.fill_between(t, ci[r'KM_estimate_lower_0.6826'], ci[r'KM_estimate_upper_0.6826'], alpha = 0.2)
            plt.xlim(0, 10)
            plt.ylim(0,1)
            sns.despine()
            plt.text(5, 0.5,f' N = {num_events} \n kon = {k:0.4f} s$^{{{-1}}}$ \n \u03C4 = {d_time:0.4f} s',ha='left', fontsize='large')
            plt.title(f'{org} -> {des}')
            plt.xlabel('time (s)')
            plt.ylabel('Cumulative Probability')
            #plt.tight_layout()
            os.makedirs(path + r'\\k', exist_ok = True)
            plt.savefig(path + f'\\k\\{org}_{des}_off.tif', dpi = 1200)
            plt.close()
            koff.append(k)
    
    
    model = Exponential.fit(x = dwell_1)
    k = model.params[0]
    print(k * d_10/(d_12+d_10))
    print(k * d_12/(d_12+d_10))
    


    print('----Done----')
    print(kon[0], '\t' , k * d_12/(d_12+d_10))
    print(k * d_10/(d_12+d_10), '\t' , koff[1])
    
    return kon, koff

path = r'H:\TIRF\20221229\lane3\dmc1\90s\FRET\1'
process(path)








