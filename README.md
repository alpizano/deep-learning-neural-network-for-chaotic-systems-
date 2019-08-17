# Deep Learning Neural Network for Chaotic Systems 

# Introduction
Hiya, this will be a guide for my senior design project that spans two semesters (Senior Design I [Spring 2019] and Senior Design II[Fall 2019]) at Purdue University - Northwest. This project involved designing and developing a neural network that would localize, object detect and predict an outcome of where a ball would land in a roulette wheel.

# Contents
0. Setup Tests
1. Prerequisite Tasks
2.
3.

# Setup Tests
These are setup test to run to make sure for example, PyTorch is utilizing your NVIDIA CUDA GPU:
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
The other setup is to make sure OpenCV is functioning with webcam.
