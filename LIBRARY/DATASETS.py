# Disclaimer:
# This file is part of the undergraduate thesis of Mr. Efthimis Michalis.
# The thesis was developed under the supervision of Assistant Prof. Aggelos
# Pikrakis, in the Department of Informatics, School of ICT, University of
# Piraeus, Greece. 

#DATASETS.py
import numpy as np
from struct import *

#### CIFAR-10
def LoadCIFAR_10(trn=50000,tsn=10000):
    def CIFAR_10SetDataANDLabels(Numb,BinFile):
        FileReader=open(BinFile,'rb')
        a=np.arange(10)
        Image=np.zeros((Numb,32,32,3))
        Label=np.zeros((Numb,10))
        i=0
        for i in range(0,Numb):
            Label[i,:]=np.sign(-np.fabs(a-FileReader.read(1)[0]))+1
            Image[i,:,:,0]=np.array([list(unpack_from("<%iB" %(1024),FileReader.read(1024)))]).reshape(32,32)
            Image[i,:,:,1]=np.array([list(unpack_from("<%iB" %(1024),FileReader.read(1024)))]).reshape(32,32)
            Image[i,:,:,2]=np.array([list(unpack_from("<%iB" %(1024),FileReader.read(1024)))]).reshape(32,32)
            Image[i,:,:,:]=Image[i,:,:,:]/255

        FileReader.close()
        return Image,Label
    ##Train
    Set=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    # 1 <= trainNumb <= 50000
    Numb=trn
    BinFile = "../DATASETS/CΗIFAR-10/DATA_BATCH_0.bin"
    I=int(np.ceil(Numb/10000))
    TrainIm=np.zeros((1,32,32,3))
    TrainLabel=np.zeros((1,10))
    for i in range(1,I+1):
        print('Read Train batch file '+str(i))
        BinFile=BinFile.replace("_"+str(i-1),"_"+str(i))
        K=min(I-i,1)
        F=10000+(Numb-10000*i)*(1-K)
        trIm,trLb=CIFAR_10SetDataANDLabels(F,BinFile)
        TrainIm=np.concatenate((TrainIm,trIm))
        TrainLabel=np.concatenate((TrainLabel,trLb))
    TrainIm=TrainIm[1:TrainIm.shape[0],:,:,:]
    TrainLabel=TrainLabel[1:TrainLabel.shape[0],:]
    # 1 <= TestNumb <= 10000
    Numb=tsn
    BinFile = "../DATASETS/CΗIFAR-10/TEST_BATCH.bin"
    print('Read Test file')
    TestIm,TestLabel=CIFAR_10SetDataANDLabels(Numb,BinFile)
    print(TrainIm.shape)
    return TrainIm,TrainLabel,TestIm,TestLabel,Set

#### MNIST ###### 60000 Trains sample 10000 Test samples
def MNISTdata(trn=60000,tsn=10000):
    def Get_From_Bin_File(Numb,LabelBin,ImageBin):
        print("Read Labels from %s.\nRead Images from %s"%(LabelBin,ImageBin))
        ImBinReader=open(ImageBin,'br')
        LabelBinRader=open(LabelBin,'br')
        ImBinReader.read(16)
        LabelBinRader.read(8)
        Image=np.zeros((Numb,28,28,1))
        Label=np.zeros((Numb,10))
        a=np.arange(10)
        for i in range(0,Numb):
            Image[i,:,:,:]=np.array([list(unpack_from("<%iB" %(784),ImBinReader.read(784)))]).reshape(28,28,1)/255
            Label[i,:]=np.sign(-np.fabs(a-LabelBinRader.read(1)[0]))+1
        ImBinReader.close()
        LabelBinRader.close()
        return Image,Label

    TrainImagePath="../DATASETS/MNIST/TrainImages.bin"
    TrainLabelPath="../DATASETS/MNIST/TrainLabels.bin"
    # 1 <= trainNumb <= 60000
    Numb=trn
    TrainIm,TrainLabel=Get_From_Bin_File(Numb,TrainLabelPath,TrainImagePath)

    TestImagePath="../DATASETS/MNIST/TestImages.bin"
    TestLabelPath="../DATASETS/MNIST/TestLabels.bin"
    # 1 <= testNumb <= 10000
    Numb=tsn
    TestIm,TestLabel=Get_From_Bin_File(Numb,TestLabelPath,TestImagePath)
    Set=["0","1","2","3","4","5","6","7","8","9"]
    return TrainIm,TrainLabel,TestIm,TestLabel,Set

## 150 Samples
def IRISdata(trn=120):
    ## 0<=TR_data<=150
    N=trn
    # Testing samples 150-N
    a=np.arange(3)
    IRIS_DATA=open('../DATASETS/IRIS/IRIS_DATA.bin','br')

    TrainIm=np.zeros((N,4))
    TrainLabel=np.zeros((N,3))
    for i in range(0,N):
        TrainIm[i]=np.array([list(unpack_from("<4f",IRIS_DATA.read(16)))])
        TrainLabel[i]=np.sign(-np.fabs(a-unpack_from("f",IRIS_DATA.read(4))[0]))+1

    TestIm=np.zeros((150-N,4))
    TestLabel=np.zeros((150-N,3))
    for i in range(0,150-N):
        TestIm[i]=np.array([list(unpack_from("<4f",IRIS_DATA.read(16)))])
        TestLabel[i]=np.sign(-np.fabs(a-unpack_from("f",IRIS_DATA.read(4))[0]))+1
    IRIS_DATA.close()
    Set=['Iris-setosa','Iris-versicolor','Iris-virginica']
    return TrainIm.reshape(N,4,1,1),TrainLabel,TestIm.reshape(150-N,4,1,1),TestLabel,Set

def CHARS74K_NUM_CAPS_FONTS(N=36576):
    Path='../DATASETS/CHARS74K_NUM_CAPS_FONTS/CHARS74K_NUM_CAPS_FONTS.bin'
    RDataSet=open(Path,'br')
    TrainIm=np.zeros((N,28,28,1))
    TrainLabel=np.zeros((N,36))
    for i in range(0,N):
        Laybel=RDataSet.read(1)[0]
        TrainLabel[i,Laybel]=1
        TrainIm[i]=np.array([list(unpack_from("<%iB" %(784),RDataSet.read(784)))]).reshape(28,28,1)/255
    RDataSet.close()
    Set=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
        'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    return TrainIm,TrainLabel,Set


# Get all MNIST CH74K_NUM_CAPS_FONTS as Training
def Merge_MNIST_CH74K_NUM_CAPS_FONTS(blender=True):
    def MergeTrain_Test(Dtr,Ltr,Dts,Lts):
        D_Merge=np.concatenate((Dtr,Dts),axis=0)
        L_Merge=np.concatenate((Ltr,Lts),axis=0)
        return D_Merge,L_Merge
    def MergeDataSets(Data_1,Label_1,Data_2,Label_2):
        D_Merge=np.concatenate((Data_1,Data_2),axis=0)
        L_Merge=np.concatenate((Label_1,Label_2),axis=0)
        return D_Merge,L_Merge

    ChDS_D,ChDS_L,ChDS_Names=CHARS74K_NUM_CAPS_FONTS()
    MNIST_Dtr,MNIST_Ltr,MNIST_Dts,MNIST_Lts,_=MNISTdata()
    MNIST_D_Merge,MNIST_L_Merge=MergeTrain_Test(MNIST_Dtr,MNIST_Ltr,MNIST_Dts,MNIST_Lts)

    MNIST_L_Merge=np.concatenate((MNIST_L_Merge,np.zeros((MNIST_L_Merge.shape[0],26))),axis=1) ## MNIST 10 classes to 36

    TrainData,TrainLabel=MergeDataSets(
                                    ChDS_D, ChDS_L,
                                    MNIST_D_Merge, MNIST_L_Merge
                         )
    if blender:
        N_tr=TrainData.shape[0]
        Rarray = np.random.choice(np.arange(0, N_tr), replace=False, size=(1, N_tr)).reshape(N_tr)
        TrainData=TrainData[Rarray]
        TrainLabel=TrainLabel[Rarray]
    return TrainData,TrainLabel,ChDS_Names
