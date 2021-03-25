# Application of Convolutional Neural Networks, Car Plate Recognition

### Python3 Project

#### Library requirements
1. matplotlib
2. numpy
3. scipy
4. skimage
5. sklearn
6. pandas
7. tensorflow

|            |       IMAGE PREPROCESSING       |   SEGMENTATION  | RECOGNITION |
| ---------- | ------------------------------- | ------------ |  ---------- |
| **INFO**   | Shadow removal, Edge detection | DBSCAN   | Deep CNN    |
|**LIBRARY** | [scikit-image](https://github.com/scikit-image)| [scikit-learn](https://github.com/scikit-learn) | [Tensorflow](https://github.com/tensorflow)  |






## Get Started
1. Clone git
```bush
$ git clone https://github.com/efthymis-mcl/Convolutional_Neural_Networks--Car_Plate_Recognition
```
2. Move to folder
```bush
$ cd path\to\Convolutional_Neural_Networks-Car_Plate_Recognition
```
3. Set up Carplates
```bush
Convolutional_Neural_Networks-Car_Plate_Recognition$ python3 setup.py
```
4. Move to SCRIPTS folder
```bush
Convolutional_Neural_Networks-Car_Plate_Recognition$ cd SCRIPTS
```
5. Here are tow files **ALL_CAR_PLATES_TESTING_RESULTS.py** and **SINGLE_CAR_PLATE_RESULT_DETAILS.py**, the second one has input arguments, e.g:
```bush
Convolutional_Neural_Networks-Car_Plate_Recognition/SCRIPTS$ python3 SINGLE_CAR_PLATE_RESULT_DETAILS.py id
```
where **id** is the number from carplate's file name.
