# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthymis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece.

# methods.py
import numpy as np
import matplotlib.pyplot as plt


def img2points(input, thres=0.45):
    points = np.where(input > thres)
    pointsy = points[1].reshape(points[1].shape[0], 1)
    pointsx = points[0].reshape(points[0].shape[0], 1)
    return np.concatenate((pointsx, pointsy), axis=1)


# draw Rects on the image
# c :rgb list [r,g,b] 0 to 1 range
def bounding_box_draw(img_gray, cornres):
    for corner in cornres:
        pos0x = corner[0]
        pos0y = corner[1]
        pos1x = corner[2]
        pos1y = corner[3]
        plt.plot(
            [pos0y, pos1y, pos1y, pos0y, pos0y],
            [pos0x, pos0x, pos1x, pos1x, pos0x],
            '--b'
        )
    img_gray = img_gray.reshape(*img_gray.shape, 1)
    img_gray = np.concatenate([img_gray]*3, axis=2)
    plt.imshow(img_gray)


