# Fiducial Marker Analysis

This repo provides a practical usage and anaylysis for the fiducial marker systems AprilTag, ArUco, ChAruCo, STag and TopoTag. Dependencies of each marker system is different, which can be checked from the detector.py. Since each system requires different enviroments, this repo also makes it possible to use them individually.


# Dataset

A dataset containing 100 images per each size and type of markers can be found in [this Google drive folder](https://drive.google.com/drive/folders/1COT8RgIYBdjq2AMysAtIkrrukDtPGlNX?usp=sharing). Just download the dataset inside of the repo and call the following function from `utils.py`

```GetImages(is_camera=use_camera, dataset_name=dataset_name)```

* where `is_camera` variable should be `False` since the dataset is used rather than the camera
* `dataset_name` is the name of the subfolder containing the desired marker and size
