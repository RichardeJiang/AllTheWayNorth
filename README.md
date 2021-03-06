# Vehicle Detection with SSD on TensorRT
This is the Python implementation for the vehicle detection project, mainly using MobileNet-SSD-V1 based on Tensorflow. The experiment is carried out on nVidia Jetson Nano development board, accelerated with the TensorRT framework, and 2 types of settings are supported:
- Video input, check vid.py;
- Live camera input, check ff.py.

The following link ([here](https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9)) is, at least to me, an excellent resource to get started on SOTA object detection algorithms (RCNN family, YOLO, SSD, etc), and will make the idea behind the whole project easier to understand.

## Libraries:
- OpenCV 3.3.1, pre-built inside the JetPack;
- numpy 1.16;
- Tensorflow 1.13, dedicated package released specially for Jetson Nano.

## Files:
The two main files follow the same logic: load the TensorRT model, take in the image from either video stream or live camera, infer, and lastly render the image output with detection labels. 

### Main use:
- vid.py: take in a video in the same folder, change the name in **line 143**; implementation mainly comes from this excellent [explanation](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/). I really owe to this one as it provides some clear insights on why we do it this way;
- ff.py: take in the live camera input, using the standard CSI Raspberry Pi v2 camera.

### Support use:
- cam.py: test code for running the camera, taken directly from [JetsonHack Tutorial](https://www.jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/).
- lane.py, lanet.py: test code for lane detection, taken directly from existing Udacity projects on Github; credits from [here](https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132) and [here](https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking/blob/master/Part%20I%20-%20Simple%20Lane%20Detection/P1.ipynb). Basic ideas are the same: transform the image to another color space, run edge detector, for all lines run the Hough Transformation to find intersection of lines, and consolidate those lines.
- wrapper.py: wrapper code for lane detection: loop through the video frames and invoke methods from other files to do the lane detection job; doing it in this way such that later we can directly import the files in ``` ff.py ``` for final product.

## Experiment Results and Remarks:
I guess this self-init project demonstrated exactly how the industry is different from course assignments. Take the Computer Vision course CS6476 as the example, all homeworks are hand-crafted by instructors and the results are well designed and could be interpreted clearly; however, in real work like this, no one knows what to expect from the device you use, the input given to you, etc, hence you need to go through all sorts of materials to find it out. It's much more painful than the homework, but somehow you learns more.

### Vehicle Detection:
Based on the experiments it can be shown that the Nano board is extremely limited in terms of computation power (although equipped with the 128-core GPU and without denying that it's much much much more superior/advanced than the most common single board computer - Raspberry Pi). Several facts:
- Due to the fact that Nano comes with the ARM processor, a lot of packages/libs have to be manually configured / compiled (or even consider other alternatives), including but not limited to Sublime Text, Caffe, and Tensorflow. 
  - Tensorflow is still fine, as you can simply wget the package and install (although the compilation time of the dedicated Tensorflow package is somewhere around 30 ~ 40 minutes);
  - scipy: crashed a few times before the success installation with ``` pip ```;
  - Caffe is already a ^&%!$#, nvm;
  - Sublime Text is not supported; used Visual Studio built by community instead.
- I wondered at the first place how large the GPU RAM is, which turns out to be "sharing the same RAM with CPU", so consider the simple setting where the Ubuntu desktop is used, we are usually left with somewhere around 900MB to 1.2GB for GPU and deep learning;
- Due to the above GPU resource limitation, the experiment facts (inference only): 
  - standard YOLO-small (Tensorflow): 25 FPS on GTX1060, crashed on Nano;
  - standard YOLO-tiny (Tensorflow): 25 ~ 30 FPS on GTX1060, single image working on Nano, video crashed;
  - SSD-MobileNet-v2 (TensorRT): crashed;
  - SSD-MobileNet-v1 (TensorRT): max 8 FPS for camera input, 13 FPS on video (reasonable as it seems that the camera takes up quite some RAM and Nano crashed the first time I booted up and configured the camera; actually considering the computation power constraint as specified above, 13 FPS is pretty impressive already).

You may check out the following image; it's a picture captured by our team when we did a real road test with the Jetson Nano setup coupled with the camera on and powered by the power bank:

<img src="./assets/use.png" width="600px">

Take note of the red marking area: the recognized vehicles are being displayed on the screen. Due to the moving car we ourselves were in (shaky and moving), it's not completely obvious.

### Lane detection:
Up till now during the summer, sad story in terms of lane detection: basic ideas from Udacity and previous CV experience all convey the same rationale we shall follow: gray scale image, edge detection, then Hough Transformation, no questions. However, the real simplest logic as written in ``` lane.py ``` is not working well: if we only use the grey scale image, then when doing the edge detection it's obvious that many lines emerge and the whole Canny result is messy, thus both lanes detected tend to be the ones on the extreme left and right, sometimes even beyond the boundary and caused overflow problems. Therefore, the use of HSV colorspace is necessary, as specified in ``` lanet.py ``` and the original [post](https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0): what we are looking for is those lane blocks / lines in **yellow** or **white**, which helps us filter out irrelevant edges efficiently.

The final combined output can be viewed below:

<img src="/assets/road.png" width="550px">

Due to the connectivity issue with the board no screen recording applications were installed on the Nano when committing the final result, hence the image was obtained by taking photo directly from the screen. However, you can still see the recognized cars and the lane currently in use.

## TODOs:
Frankly speaking, what has been accomplished here is a tiny section of what should be presented as the final result. It's rather obvious that sometimes doing deep learning doesn't require rich knowledge of ML, which is demonstrated precisely by the progress up till now; to put it straightforward, the above image could only be regarded as a merely justifiable (if not awkward) imitation of the Udacity project. Hence, to truly understand the idea behind deep learning and the training process, the following should be carried out as the ultimate goal:
- Get a more profound understanding on the training and labelling processes for object detection in DL;
- Label our own data, or obtain the labeled data online, and further train the current SSD network;
- Get hands on the nVidia DIGITS lib for more sophisticated self-driving car training platform.
