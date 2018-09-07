import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image

image = Image.open('image.png')
#image.show()
img=np.asarray(image)


segments_slic = slic(img, n_segments=200, compactness=10, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
plt.imshow(mark_boundaries(img, segments_slic))
plt.show()
