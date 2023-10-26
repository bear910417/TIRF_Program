from Image_Loader import Image_Loader
from aoi_utils import cal
import numpy as np

path = r'H:\TIRF\20230728\lane4\dmc1\20s'
mpath = r'H:\TIRF\20230501 mapping\mapping 5'
g_length, r_length, b_length, g_start, r_start, b_start = cal(path)
im = Image_Loader(0, 9, path, g_length, r_length, b_length, 0, 0, 0, 0)

im.load_image()
im.gen_dimg(anchor = 0, mpath = mpath, maxf = 420, minf = 178, channel = 'green', plot = False)
coord_list = im.det_blob()
coord_list = np.array(coord_list)
print(coord_list[10:15])
print('-------')

#