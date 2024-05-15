import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import tifffile as tf



np.set_printoptions(threshold=sys.maxsize)

path = r'H:\idl_test'



# array = np.fromfile(path + r'\tif.tif', dtype=(np.dtype('<u1'))) 
# print(array.shape)
#print(np.fromfle(path + r'\tif.tif', dtype=(np.dtype('<u4')), offset = 1575954+118, count = 1))
im = tf.imread(path + r'\DNA2.tif')
im = np.average(im, axis = 0 )
tf.imshow(im, vmin = np.min(im), vmax = np.max(im), cmap = 'gray')

# trace = np.fromfile(path + r'\DNA1.traces', dtype=(np.dtype('>i2'), (10, 336)), offset = 5).reshape((10, 336))
# r = trace[:, ::2]
# g = trace[:, 1::2]
# print(trace.shape)

# plt.plot(np.arange(10), (r)[:, 90])

plt.show()