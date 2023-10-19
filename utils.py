import numpy as np
import os
import time as rtime
from scipy.ndimage import uniform_filter1d

def uf(t, lag, axis = -1):
    return uniform_filter1d(t, size = lag, mode = 'nearest', axis = axis)

def update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, bkps, lag, show):
    mode_dict = {
        0 : 'lines',
        1 : 'markers'
    }
    try:
        hist_range = (relayout['xaxis.range[0]'], relayout['xaxis.range[1]'])
    except:
        hist_range = (0, np.inf)

    if np.any(fret_b):
        uf_time_b = uf(time['b'], lag)
        fig.update_traces(x = uf_time_b, y = uf(fret_b[i], lag), mode = mode_dict[scatter], visible = ('FRET BG' not in show), selector = dict(name='fret_b'))
        fig.update_traces(x = uf_time_b, y = uf(bb[i], lag), mode = mode_dict[scatter], visible = ('BB' not in show), selector = dict(name='bb'))
        fig.update_traces(x = uf_time_b, y = uf(bg[i], lag), mode = mode_dict[scatter], visible = ('BG' not in show), selector = dict(name='bg'))
        fig.update_traces(x = uf_time_b, y = uf(br[i], lag), mode = mode_dict[scatter], visible = ('BR' not in show), selector = dict(name='br'))
        
        fig.update_traces(x = uf_time_b, y = uf(bb[i]+ bg[i]+ br[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot B' not in show), selector = dict(name='tot_b'))

        fig.update_traces(x = [x[1] for x in bkps['b'][i]], y = uf(bb[i], lag)[[y[0] for y in bkps['b'][i]]], mode = 'markers', selector = dict(name='b_bkps'))
        fig.update_traces(x = [x[1] for x in bkps['fret_b'][i]], y = uf(fret_b[i], lag)[[y[0] for y in bkps['fret_b'][i]]], mode = 'markers', selector = dict(name='fret_b_bkps'))
        
        if ('Tot B'  in show):
            fig.update_layout(yaxis4 = dict(range = (0, np.max(np.concatenate((bb[i], bg[i], br[i])))))),
        else:
            fig.update_layout(yaxis4 = dict(range = (0, np.max((bb[i] + bg[i] + br[i]))))),
        
        hfilt_b = (uf_time_b > hist_range[0]) * (uf_time_b < hist_range[1])
        
        fig.update_traces(y = uf(fret_b[i], lag)[hfilt_b], selector = dict(name='Histogram_b'))
      
   
    if np.any(fret_g):
        uf_time_g = uf(time['g'], lag)
        fig.update_traces(x = uf_time_g, y = uf(fret_g[i], lag), mode = mode_dict[scatter], visible = ('FRET GR' not in show), selector = dict(name='fret_g'))
        fig.update_traces(x = uf_time_g, y = uf(gg[i], lag), mode = mode_dict[scatter], visible = ('GG' not in show), selector = dict(name='gg'))
        fig.update_traces(x = uf_time_g, y = uf(gr[i], lag), mode = mode_dict[scatter], visible = ('GR' not in show), selector = dict(name='gr'))
        fig.update_traces(x = uf_time_g, y = uf(gg[i]+ gr[i], lag), mode = mode_dict[scatter], line = dict(dash = "longdash", width = 2), visible = ('Tot G' not in show), selector = dict(name='tot_g'))

        fig.update_traces(x = [x[1] for x in bkps['g'][i]], y = uf(gg[i], lag)[[y[0] for y in bkps['g'][i]]], mode = 'markers', selector = dict(name='g_bkps'))
        fig.update_traces(x = [x[1] for x in bkps['fret_g'][i]], y = uf(fret_g[i], lag)[[y[0] for y in bkps['fret_g'][i]]], mode = 'markers', selector = dict(name='fret_g_bkps'))
        
        if ('Tot G' in show):
            fig.update_layout(yaxis3 = dict(range = (0, np.max(np.concatenate((gg[i], gr[i]))))))
        else:
            fig.update_layout(yaxis3 = dict(range = (0, np.max(gg[i] + gr[i]))))

        
        hfilt_g = (uf_time_g > hist_range[0]) * (uf_time_g < hist_range[1])       
        fig.update_traces(y = uf(fret_g[i], lag)[hfilt_g], selector = dict(name='Histogram_g'))

    if np.any(rr):
        fig.update_traces(x = uf(time['r'], lag), y = uf(rr[i], lag), mode = mode_dict[scatter], visible = ('RR' not in show), selector = dict(name='rr'))
        fig.update_traces(x = [x[1] for x in bkps['r'][i]], y = uf(rr[i], lag)[[y[0] for y in bkps['r'][i]]], mode = 'markers', selector = dict(name='r_bkps'))
        if not np.any(fret_g):
            fig.update_layout(yaxis3 = dict(range = (0, np.max(rr[i])))),

    
    if np.any(fret_b) or np.any(fret_g) or np.any(rr):
        fig.update_layout(xaxis1 = dict(range=(min(time['g'][0], time['b'][0], time['r'][0]), max(time['g'][-1], time['b'][-1], time['r'][-1]))))


    return fig

def change_trace(changed_id, event, i, N_traces, fig):
    if ('next' in changed_id) or (('n_events' in changed_id) and event['key'] == 'w'):
        if i < N_traces-1:
            i = i+1
            fig.layout.shapes=[]
            #fig = draw(fig, tot_bkps, i, time_g, dead_time, color, total_frame)
           
    elif ('previous' in changed_id)  or (('n_events' in changed_id) and event['key'] == 'q'):
        if i>0:
            i = i-1
            fig.layout.shapes=[]
            #fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
              
    elif 'tr_go' in changed_id:
       
        if i.isdigit():
            if int(i) < N_traces:
                i = int(i)
            else:
                i = 0
        else:
            i = 0
        fig.layout.shapes=[]
    
    return i, fig

def select_good_bad(changed_id, event, i, select_list_g):
    
    
    if ('set_good' in changed_id) or (('n_events' in changed_id) and event['key'] == 'z'):  
        if select_list_g[i] == 1:
            select_list_g[i] = 0
        else:
            select_list_g[i] = 1
        
    if ('set_bad' in changed_id ) or (('n_events' in changed_id) and event['key'] == 'x'):  
        if select_list_g[i] == -1:
            select_list_g[i] = 0
        else:
            select_list_g[i] = -1
         
    return select_list_g

def render_good_bad(i, select_list_g):
    white_button_style = {'background-color': '#f0f0f0', 'color': 'black'}
    red_button_style = {'background-color': 'red', 'color': 'white'}                        
    green_button_style = {'background-color': 'green', 'color': 'white'}

    if select_list_g.shape[0] == 0:
        good_style = white_button_style
        bad_style = white_button_style
        return good_style, bad_style

    if select_list_g[i] == 1:
        good_style = green_button_style
        bad_style = white_button_style
    elif select_list_g[i] == 0:
        good_style = white_button_style
        bad_style = white_button_style
    else:
        good_style = white_button_style
        bad_style = red_button_style
    
    return good_style, bad_style


def breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps):
    trans = {
        0 : 'fret_g',
        1 : 'fret_b',
        2 : 'b',
        3 : 'b',
        4 : 'b',
        5 : 'b',
        6 : 'g',
        7 : 'g',
        8 : 'g',
        9 : 'r', 
        10 : 'fret_g',
        11 : 'fret_b',
        12: 'b',
        13: 'g',
        14: 'r',
    }

    if 'dtime' in changed_id and channel != None:
        if mode == 'Add':

            bkps[channel][i].append((0, time[channel][0]))
            bkps[channel][i] = sorted(bkps[channel][i])                   

        elif mode == 'Remove':
     
            try:
                bkps[channel][i].pop(0)
                bkps[channel][i] = sorted(bkps[channel][i]) 
            except:
                pass
    
    if 'etime' in changed_id and channel != None:
        
        if mode == 'Add':

            bkps[channel][i].append((time[channel].shape[0]-1, time[channel][-1]))
            bkps[channel][i] = sorted(bkps[channel][i])                   

        elif mode == 'Remove':
     
            bkps[channel][i].pop(-1)
            bkps[channel][i] = sorted(bkps[channel][i]) 


    if 'graph.clickData' in changed_id:
        if isinstance(clickData, dict):
            c_num = clickData["points"][0]["curveNumber"]
            channel = trans[c_num]

            if mode == 'Add':
                if c_num <10:
                    idx = clickData["points"][0]["pointNumber"]
                    idx_t = time[channel][idx]
                    bkps[channel][i].append((idx, idx_t))
                    bkps[channel][i] = sorted(bkps[channel][i])  
                    #fig.layout.shapes=[]
                    #fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)  
                            
            elif mode == 'Remove':
                if 10 <= c_num <=14:
                    rem_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i].pop(rem_idx)                
                    # fig.layout.shapes=[]
                    # fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame) 
                    
            elif mode == 'Except':
                if 10 <= c_num <=14:
                    exp_idx = clickData["points"][0]["pointNumber"]
                    bkps[channel][i] = [bkps[channel][i][exp_idx]]      
                    mode = "Add"
                    # fig.layout.shapes=[]
                    # fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
    
                    
    if mode == 'Clear':
        mode = "Add"
        if channel != None:
            bkps[channel][i] = []
        #fig.layout.shapes=[]

    if mode == 'Clear All':
        for channel in bkps:
            bkps[channel][i] = []
        #fig.layout.shapes=[]
        mode = "Add"
        
    if mode == 'Set All':
        mode = "Add"
        if channel != None:
            for keys in bkps:
                bkps[keys][i] = bkps[channel][i]

    return bkps, mode

def sl_bkps(changed_id, path, bkps, mode):
    
    if ('save_bkps' in changed_id) or (mode == 'Clear All'):
        
        # if not os.path.exists(path+r"/images"):
        #     os.mkdir(path+"/images")
        # fig.write_image(path+f"/images/trace{i}.png", engine="kaleido",width=1600,height=800,scale=10)
        mode = 'Add'
        try:
            bkps_bac = np.load(path+r'/breakpoints.npz', allow_pickle=True)
            seconds = rtime.time()
            t = rtime.localtime(seconds)
            np.savez(path+f'/breakpoints_backup_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.npz', bkps_bac)
        except:
            print('No existing save file found.')
        
        for key in bkps:
            bkps[key] = np.array(bkps[key], dtype = object)

        print(path)
        np.savez(path+r'/breakpoints.npz', **bkps)
        print('file_saved')
        
    if 'load_bkps' in changed_id:
            try:
                bkps = dict(np.load(path+r'/breakpoints.npz', allow_pickle=True))
            except:
                print('File not found')
    return bkps

def show_blob(blobs, fig_blob, smooth, time, i, hoverData):
    if blobs == None or hoverData == None:
        return fig_blob
    
    t = hoverData["points"][0]["x"]
    b = uf(blobs['b'], smooth, 2)
    g = uf(blobs['g'], smooth, 2)
    r = uf(blobs['r'], smooth, 2)

    minf = int(blobs['minf'])
    maxf = int(blobs['maxf'])
    
    z_list = []
    if np.any(b):
        uf_time_b = uf(time['b'], smooth)
        x = np.abs(uf_time_b - t).argmin()
        z_list = z_list + [b[i][0][x], b[i][1][x], b[i][2][x]]
    else:
        z_list = z_list + [np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9))]

    if np.any(g):
        uf_time_g = uf(time['g'], smooth)
        x = np.abs(uf_time_g - t).argmin()
        z_list = z_list + [g[i][0][x], g[i][1][x]]
    else:
        z_list = z_list + [np.zeros((9, 9)), np.zeros((9, 9))]

    if np.any(r):
        uf_time_r = uf(time['r'], smooth)
        x = np.abs(uf_time_r - t).argmin()
        z_list = z_list + [r[i][0][x]]
    else:
        z_list = z_list + [np.zeros((9, 9))]

    z = np.concatenate(z_list, axis = 1)


    fig_blob.update_traces(zmax = maxf, zmin = minf, selector = dict(type = 'heatmap')) 
    fig_blob['data'][0]['z'] = z
    fig_blob['layout']['coloraxis']['cmax'] = maxf
    fig_blob['layout']['coloraxis']['cmin'] = minf

    return fig_blob




        
    
                 