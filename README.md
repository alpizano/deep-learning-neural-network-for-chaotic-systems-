# Deep Learning Neural Network for Chaotic Systems 

# Introduction
Hiya, this will be a guide for my senior design project that spans two semesters (Senior Design I [Spring 2019] and Senior Design II[Fall 2019]) at Purdue University - Northwest. This project involved designing and developing a neural network that would localize, object detect and predict an outcome of where a ball would land in a roulette wheel.

# Contents
0. Setup Tests
1. Prerequisite Tasks
2.
3.

# Setup Tests
These are simple setup procedures that aid in later creating the neural network.

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
Often, we have to capture live stream with camera. OpenCV provides a very simple interface to this:
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

# Prerequisite Tasks
