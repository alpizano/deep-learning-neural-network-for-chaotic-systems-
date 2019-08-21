# Deep Learning Neural Network for Chaotic Systems 
![deep learning is awesome!](https://github.com/alpizano/Deep-Learning-Neural-Network-for-Chaotic-Systems-/blob/master/read_cover.jpg)

# Introduction
Hiya, this will be a guide for my senior design project that spans two semesters (Senior Design I [Spring 2019] and Senior Design II[Fall 2019]) at Purdue University - Northwest. This project involved designing and developing a neural network that would localize, object detect and predict an outcome of where a ball would land in a roulette wheel.

# Contents
0. Setup Tests
1. Prerequisite Tasks
2. YOLOv3
3. MTurk Bounding Box Utils

# 0. Setup Tests
These are simple setup procedures that aid in later creating the neural network.

Much of this code was written in Python, using Jupyter notebook to visualize the code or an IDE like PyCharm will suffice. Jupyter notebook can be installed via Anaconda here:
https://www.anaconda.com/distribution/#download-section

Let's make sure PyTorch is utilizing your NVIDIA CUDA GPU (if you have one that is):
```
In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'GeForce GTX 950M'

In [6]: torch.cuda.is_available()
Out[6]: True
```
Often, we have to capture live stream with camera. OpenCV provides a very simple interface to this. Further more information, please refer to this website:

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
```
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

# 1. Prerequisite Tasks
In the first semester of Senior design, our group was led by Dr. Bin Chen. We were given some prelimary tasks to complete before embarking on actually trying to get a localization or classifcation neural network built. The first task was an image processing task which consisted becoming used to some of the basic techniques of image processing, like utilizing the PyTorch tensors,and the pillow (PIL) libraries:
https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html

The image processing techniques mainly background subtraction and array manipulation and were written inside Jupyter notebook using Python 3. The techniques can be view below in the Jupyter notebook file:

## Background Subtraction and Array Manipulation

[Background Substraction and Array Manipulation.pynb](1.%20Prerequisite%20Tasks/Image%20Processing/background_subtraction_array_manipulation.ipynb)

## Binary Classifier
The second task we were given was to begin to implement a simple binary classifcation neural network to classify cats and dogs. Most image classification projects utilize convolution neural networks. This dog and cats dataset was provided via the Kaggle website that contains 25,000 images and can be downloaded from here:
https://www.kaggle.com/c/dogs-vs-cats/data

Below, you will find the Jupyter notebook for this image classifier and the output matlab plots:
[Dogs and Cats Binary Classifier.pynb](1.%20Prerequisite%20Tasks/Image%20Processing/background_subtraction_array_manipulation.ipynb)

# 2. YOLOv3
So this is where most of the fun begins. We utilized the YOLOv3 network, which is currently one of the fastest networks for object detection:
https://pjreddie.com/darknet/yolo/

# 3. MTurk Bounding Box Utils




