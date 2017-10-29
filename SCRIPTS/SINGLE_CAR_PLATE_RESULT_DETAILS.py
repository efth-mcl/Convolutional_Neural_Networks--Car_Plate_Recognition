# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthimis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece. 

# SINGLE_CAR_PLATER_ESULL_DETAILS.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as spm

import sys
from  METHODS import *
sys.path.append('../LIBRARY')
from NEURAL_NETWORK_TOOL import *

####################

###################

cutx=1 #cut image bottom,top pixels
cuty=1 #cut image left,right pixels

# We use MNIST-CHARS74K_NUM_CAPS_FONTS dataset gray images <28,28,1> and 36 class

Set2=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
        'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

Dir="36ClassSet"

Net = NNtool(Dir,False)
Net.SetSession()

ip=sys.argv[1]

filename="plates2/plate-"+str(ip)
img=mpimg.imread('../Files/'+filename)
img=img[cutx:-cutx,cuty:-cuty]

Per=(383*900/img.size)**(1/2)
img=spm.imresize(img,(int(img.shape[0]*Per),int(img.shape[1]*Per)),'bicubic')
plt.imshow(img)
plt.axis('off')
plt.show()

if(len(img.shape)==3):
    if(np.max(img)>1):
        img=img
    img=img[:,:,0]*1/3 + img[:,:,1]*1/3 + img[:,:,2]*1/3

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

ip=int(ip)
img=img.astype(int)
if ip==13:
    ip=1

HEinput=[[70,170,170,255,5,50],
        [50,160,170,255,5,50], #plate13


# defult default threshold
# T0=55
# T1=160
# NT0=160
# NT1=240
# dark0=0
# dark1=55
# imgH=HE(img,T0,
#             T1,
#             NT0,
#             NT1,
#             dark0,
#             dark1
#         )

imgH=HE(img,
            HEinput[ip][0],
            HEinput[ip][1],
            HEinput[ip][2],
            HEinput[ip][3],
            HEinput[ip][4],
            HEinput[ip][5]
        )

imgH=imgH/255
Eimg,thita=Cany_skimage(imgH,3,0.1,0.3)

plt.subplot(2,1,1)
plt.title('Αρχική Εικόνα')
plt.axis('off')
plt.imshow(img,cmap='Greys_r')
plt.subplot(2,1,2)
plt.title('Μετα Την Ισοστάθμιση Ιστογράμματος')
plt.axis('off')
plt.imshow(imgH,cmap='Greys_r')
plt.show()
img=np.array(imgH)
del imgH

print(img.shape)
print(img.size)
#sys.exit()

# Operator


plt.subplot(2,1,1)
plt.axis('off')
plt.imshow(Eimg,cmap='Greys_r')
plt.title('Cany Opertor',fontsize=16)

plt.subplot(2,1,2)
plt.axis('off')
plt.imshow(thita,cmap='Greys_r')
plt.title('Angle Space',fontsize=16)
plt.show()



Pon=GetPoints(Eimg) #get points with non zero value

#############
ImPlace=np.zeros((img.shape[0],img.shape[1],3))

ImPlace[:,:,0]=np.array(img)
ImPlace[:,:,1]=np.array(img)
ImPlace[:,:,2]=np.array(img)

RectDB=DBs(Pon)

RectHC=heightCluster(RectDB)

Rect=AreaCluster(RectHC,img.shape)

plt.subplot(1,3,1)
plt.axis('off')
plt.title('DBSCAN')
RectsDrawing(ImPlace, RectDB, [1,0,0])

plt.subplot(1,3,2)
plt.title('HeightCluster')
plt.axis('off')
RectsDrawing(ImPlace, RectHC, [1,0,0])

plt.subplot(1,3,3)
plt.title('AreaCluster')
plt.axis('off')
RectsDrawing(ImPlace, Rect, [1,0,0])
plt.show()
del RectDB,RectHC


#############
scaleIM=28
Imd=1 #image dimensions gray->1 rgb->3
n_clusters_=Rect.shape[0]
gap=2 #add black square around the image before scaling
Net.Initialize_Vars()
for I in range(0,n_clusters_):
    #############
    ### CNN
    pos0x=Rect[I][0]-1
    pos0y=Rect[I][1]-1
    pos1x=Rect[I][2]+1
    pos1y=Rect[I][3]+1

    Rect1=Eimg[pos0x:pos1x,pos0y:pos1y]
    thita1=thita[pos0x:pos1x,pos0y:pos1y]
#  fill edge binary images based on thita
    for i in range(Rect1.shape[0]):
        for j in range(Rect1.shape[1]):
            xstep=0
            ystep=0
            if thita1[i,j]!=0:
                if thita1[i,j]==1:
                    xstep+=-1
                elif thita1[i,j]==2:
                    xstep+=-1
                    ystep+=-1
                elif thita1[i,j]==3:
                    ystep+=-1
                elif thita1[i,j]==4:
                    xstep+=1
                    ystep+=-1
                elif thita1[i,j]==-1:
                    xstep+=1
                elif thita1[i,j]==-2:
                    xstep+=1
                    ystep+=1
                elif thita1[i,j]==-3:
                    ystep+=1
                elif thita1[i,j]==-4:
                    xstep+=-1
                    ystep+=1
                ix=i+xstep
                jy=j+ystep
                if not(ix+xstep>=thita1.shape[0] or ix+xstep<0 or  jy+ystep<0 or jy+ystep>=thita1.shape[1]):
                    while thita1[ix+xstep,jy+ystep]==0 and thita1[ix,jy+ystep]==0 and thita1[ix+xstep,jy]==0:
                        if Rect1[ix,jy]==1:
                            break
                        Rect1[ix,jy]=1
                        ix+=xstep
                        jy+=ystep
                        if ix+xstep>=thita1.shape[0] or ix+xstep<0 or  jy+ystep<0 or jy+ystep>=thita1.shape[1]:
                            break


####### Rectangle image to square image adding black columns and rows, for better scaling
    dx=pos1x-pos0x
    dy=pos1y-pos0y
    a=abs(dx-dy)/2
    z1=int(a)
    z0=int(a)+int(a!=int(a))
    maxD=dx
    if(dx>dy):
        Rect1=np.concatenate((np.zeros((dx,z0)),Rect1,np.zeros((dx,z1))),axis=1)
    elif(dx<dy):
        Rect1=np.concatenate((np.zeros((z0,dy)),Rect1,np.zeros((z1,dy))),axis=0)
        maxD=dy
    Rect1=np.concatenate((np.zeros((maxD,gap)),Rect1,np.zeros((maxD,gap))),axis=1)
    Rect1=np.concatenate((np.zeros((gap,maxD+2*gap)),Rect1,np.zeros((gap,maxD+2*gap))),axis=0)


###########

    ScalePic=ScaleArray(Rect1,scaleIM-2*gap)
    ScalePic=np.concatenate((np.zeros((scaleIM-2*gap,gap)),ScalePic,np.zeros((scaleIM-2*gap,gap))),axis=1)
    ScalePic=np.concatenate((np.zeros((gap,scaleIM)),ScalePic,np.zeros((gap,scaleIM))),axis=0)

    Score  = Net.Layers[-1].eval(feed_dict = {
             Net.Layers[0]:ScalePic.reshape(1,scaleIM,scaleIM,Imd),
             Net.keep_prob :np.ones((Net.DroupoutsProbabilitys.shape[0]))
    })

    maxScore=np.max(Score)
    plt.subplot(int(np.ceil(n_clusters_/5)),min(n_clusters_,5),I+1)

    plt.title("Max Score = "+str(maxScore)+"\nIn class : "+str(Set2[np.where(Score.reshape(36)==maxScore)[0][0]]))
    plt.axis('off')
    plt.imshow(ScalePic,cmap='Greys_r')
plt.show()