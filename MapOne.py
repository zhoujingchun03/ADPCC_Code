import numpy as np

from getOneChannelMax import getMaxChannel


def max_R(img,blockSize):
    img = img[:,:,2]
    R_map  = getMaxChannel(img, blockSize)
    R_map = R_map/255
    return R_map

