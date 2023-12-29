from Image_Loader import Image_Loader
from aoi_utils import cal
import numpy as np
from scipy.ndimage import uniform_filter1d as uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
import time

path = r'H:\TIRF\20231027\lane5\RPA\3mins'
mpath = r'H:\TIRF\20230913 mapping\1'
g_length, r_length, b_length, g_start, r_start, b_start = cal(path)
im = Image_Loader(0, 9, path, g_length, r_length, b_length, 0, 0, 0, 0)
im.load_image()
im.gen_dimg(anchor = 0, mpath = mpath, maxf = 420, minf = 178, laser = 'green', plot = False)
blob_list = im.det_blob()
coord_list = im.cal_drift(blob_list, laser = 'green', use_ch = 'all', n_slices = 4)

trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i = im.cal_intensity(coord_list)


# trace_gg, trace_gr, trace_rr, trace_bb, trace_bg, trace_br, i = im.cal_intensity(coord_list, maxf = 420, minf = 178, fsc = None)

# #

# def uf(arr, size):
#     arr = np.pad(arr, (size - 1, 0), 'reflect')
#     #arr = np.repeat(arr, size, axis = 0)
#     arr = sliding_window_view(arr, size, axis = 0)
#     arr = np.average(arr, axis = 1)
    

#     return arr

# arr = np.arange(0, 3600)
# start = time.time()
# end = time.time()
# duration = end - start
# print(duration)