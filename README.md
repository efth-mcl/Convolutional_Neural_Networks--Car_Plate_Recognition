# Application of Convolutional Neural Networks, Car Plate Recognition


|            |       IMAGE PREPROCESSING       |   SEGMENTATION  | RECOGNITION |
| ---------- | ------------------------------- | ------------ |  ---------- |
| **INFO**   | Edge detection | DBSCAN   | Deep CNN    |
|**LIBRARY** | [scikit-image](https://github.com/scikit-image)| [scikit-learn](https://github.com/scikit-learn) | [tensorflow](https://github.com/tensorflow)  |


## Set up steps
1. Run setup.sh (Linux)
    ```bush
    sh setup.sh 
    ```
    or
   1.
    ```bush
        pip install -r requirements.txt
    ```
   2.
    create "plates" folder on root of repo and download car plates from car-plates-urls.txt file. Use *wget* command
    such as:
    ```bush
        wget -P plates <car-plate-url>
    ```

## Notebooks
- [single plate test (analytical)](cpr_default/scripts/single_plate_analytical_test.ipynb)
- [performance recognition test](cpr_default/scripts/perfomance_recognition_test.ipynb)

