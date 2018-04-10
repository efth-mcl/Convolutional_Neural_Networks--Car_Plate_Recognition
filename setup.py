import urllib.request
import numpy as np
import matplotlib.image as mpimg
import os
from PIL import Image

print('Download car-plates')
os.system('mkdir PLATES')
F = open('car-plates urls', 'r')
i = 1
for f in F:

    print("plate-"+str(i)+" URL --> "+f)
    try:
        urllib.request.urlretrieve(
            f,
            "PLATES/plate-"+str(i)
        )
    except Exception as e:
        er_m = 'not found plate-'+str(i)
        print('{0}\n{1}\n{2}'.format('#'*len(er_m), er_m, '#'*len(er_m)))
    i += 1
if os.path.isfile('PLATES/plate-13'):
    img13 = mpimg.imread('PLATES/plate-13')
    A = img13[:, 0]
    w = np.where(A != [255, 255, 255])
    Min = w[0].min()
    Max = w[0].max()
    img13 = img13[Min+2:Max, :]

    img = Image.fromarray(img13, 'RGB')
    img.save('PLATES/plate-13.jpg')

    os.remove('PLATES/plate-13')
    os.system('mv PLATES/plate-13.jpg PLATES/plate-13')
