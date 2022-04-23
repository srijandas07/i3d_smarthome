# i3d_smarthome
Pytorch I3D implementation on Toyota Smarthome Dataset

This repository provides a baseline I3D training/testing code on Toyota Smarthome dataset (trimmed version).

* The data can be downloaded by request from https://project.inria.fr/toyotasmarthome/
* The videos are decoded into frames at 30 fps in a directory, say 'data'

## Training 
* Run script.sh with arguments EXPERIMENT_NAME PROTOCOL (CS, CV1, CV2) DATA_PATH
```
./script.sh test CS data
```

## Testing
* Run script_test.sh with arguments PATH_OF_THE_TRAINED_MODEL DATA_PATH
```
./script_test.sh CS rgb_SH_CS.pt
```
While testing, one can skip executing makecsv.py since the generated files are already provided in the labels directory.

## Pre-trained weights
pre-trained weights of i3d on Protocol CS and CV2 is provided in the models directory.
Difference in testing results may arise due to discripency between the tested images.
We pre-process all the images with human bounded cropping using SSD. 
However, with random cropping while training, and testing on center-crops, the classification accuracy will be higher than reported.

## Citing the dataset
    @misc{Das_2019_ICCV,
    author = {Das, Srijan and Dai, Rui and Koperski, Michal and Minciullo, Luca and Garattoni, Lorenzo and Bremond, Francois and Francesca, Gianpiero},
    title = {Toyota Smarthome: Real-World Activities of Daily Living},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}}
