# Fiducial Marker Analysis

This is a **python** repo that provides one-line **detection** and if desired **visualization** and **analysis** of several fiducial marker system both real-time([pyrealsense2](https://pypi.org/project/pyrealsense2/) with a Intel Realsense camera) or offline([dataset](https://drive.google.com/drive/folders/1COT8RgIYBdjq2AMysAtIkrrukDtPGlNX?usp=sharing)). Fiducial marker systems that are included are: AprilTag, ArUco, ChAruCo, STag, and TopoTag.

# Setup

Corresponding libraries for each fiducial marker system can be found in the `libraries` folder. Keep in mind that you **only** need to build the libraries that **you are going to use**.

# Usage

Call the function `fiducial_markers` function in the `main.py`.

## Dataset

A dataset containing 100 images for each size and type of marker can be found in [this Google drive folder](https://drive.google.com/drive/folders/1COT8RgIYBdjq2AMysAtIkrrukDtPGlNX?usp=sharing). Just download the dataset inside of the repo and call the following function from `utils.py`

```GetImages(is_camera=use_camera, dataset_name=dataset_name)```

* where `is_camera` variable should be `False` since the dataset is used rather than the camera
* `dataset_name` is the name of the subfolder containing the desired marker and size

# Contact

If you find any bugs or have any questions about the code, please report to the Issues page.