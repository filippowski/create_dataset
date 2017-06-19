from util import get_points_from_json, put_points_in_json
import os
from skimage import io

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#
# %matplotlib inline
plt.rcParams["figure.figsize"] = (20,20)

from PIL import Image, ImageDraw
import itertools
from itertools import islice, count


path_to_bunch = '/home/filippovski/deep-learning/MULTITASK/CREATE/bunch000/667.obj'

import re

path2img = os.path.join(path_to_bunch, '0.jpg')
path2json = re.sub(".jpg", ".json", path2img)

print path2json

img = Image.open(path2img)

draw = ImageDraw.Draw(img)

pts = get_points_from_json(path2json)
pts = (pts+1)*(0.5*img.height)
print pts

draw.point(list(pts))
plt.imshow(img)

plt.show()