# Vehicle Detection
This is the Python implementation for the vehicle detection project, mainly using MobileNet-SSD-V1 based on Tensorflow. The experiment is carried out on nVidia Jetson Nano development board, accelerated with the TensorRT framework, and 2 types of settings are supported:
- Video input, check vid.py;
- Live camera input, check ff.py.

## Libraries:
- OpenCV 3.3.1, pre-built inside the JetPack;
- numpy 1.16;
- Tensorflow 1.13, special package released by Google for Jetson Nano.

## Files:
The two main files follow the same logic: load the TensorRT model, take in the image from either video stream or live camera, infer, and lastly render the image output with detection labels. 
- vid.py: take in a video in the same folder, change the name in **line 143**;
- ff.py: take in the live camera input, using the standard CSI Raspberry Pi v2 camera;
- cam.py: test code for running the camera.

## Remarks:
Based on the experiments it can be shown that the Nano board is extremely limited in terms of computation power.
