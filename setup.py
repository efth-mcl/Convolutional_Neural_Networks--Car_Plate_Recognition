import urllib.request
import numpy as np
import matplotlib.image as mpimg
import os


F=open('car-plates urls','r')
i=1
for f in F :
    urllib.request.urlretrieve(
        f,
         "plate-"+str(i)
    )
    i+=1


img13=mpimg.imread('plate-13')
A=img13[:,0]
w=np.where(A!=[255,255,255])
print(w[0])
Min=w[0].min()
Max=w[0].max()
print(Min)
img13=img13[Min+2:Max,:]
from PIL import Image

img = Image.fromarray(img13, 'RGB')
img.save('plate-13.jpg')

os.remove('plate-13')
os.system('mv plate-13.jpg plate-13')
os.system('mkdir PLATES')
os.system('mv plate* PLATES')
