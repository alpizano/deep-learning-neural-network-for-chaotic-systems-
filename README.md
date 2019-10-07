# Deep Learning Neural Network for Chaotic Systems 
![deep learning is awesome!](https://github.com/alpizano/Deep-Learning-Neural-Network-for-Chaotic-Systems-/blob/master/read_cover.jpg)

# Introduction
Hiya, this will be a guide for my senior design project that spans two semesters Senior Design I (Spring 2019), and Senior Design II (Fall 2019), at Purdue University - Northwest. This project involved designing and developing a neural network that would localize, object detect and predict an outcome of where a ball would land in a roulette wheel.

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

## Background Subtraction and Array Manipulation
The image processing techniques mainly background subtraction and array manipulation and were written inside Jupyter notebook using Python 3. The techniques can be view below in the Jupyter notebook file:

[Background Substraction and Array Manipulation.pynb](1.%20Prerequisite%20Tasks/Image%20Processing/background_subtraction_array_manipulation.ipynb)

## Binary Classifier
The second task we were given was to begin to implement a simple binary classifcation neural network to classify cats and dogs. Most image classification projects utilize convolution neural networks. This dog and cats dataset was provided via the Kaggle website that contains 25,000 images and can be downloaded from here:

https://www.kaggle.com/c/dogs-vs-cats/data

Below, you will find the Jupyter notebook for this image classifier and the output matlab plots:

[Dogs and Cats Binary Classifier.pynb](1.%20Prerequisite%20Tasks/Image%20Processing/background_subtraction_array_manipulation.ipynb)

# 2. YOLOv3
![You Only Look Once!](https://github.com/alpizano/Deep-Learning-Neural-Network-for-Chaotic-Systems-/blob/master/darknet_ss.png)
So this is where most of the fun begins. We utilized the YOLOv3 network, which is currently one of the fastest networks for object detection:

https://pjreddie.com/darknet/yolo/

# 3. MTurk Bounding Box Utils
When acquiring data for the neural network, we searched youtube and google for various different images of roulette wheels/balls to train the network. We needed to train the network to detect the ball, 0 fret, and center of the wheel, so we need to feed the correct data with bounding boxes to train our network, essentially feeding it the correct answer so it could learn. We utilized Amazon's Mechnical Turk utility were we submit the data and the workers process the jobs to annotate your images. We have over 1,000 images, so we wrote a Python script to take the .CSV file that MTurk sends up with the bouding box data specificed as an XML string in one of the columns, and we parsed that data out and were able to visualize the bouding box via the matplotlib library.

The full script can be found here:

[MTurk Annotation Viewer.pynb](https://github.com/alpizano/mturk_bbutils/blob/5a94cb021b7b2185524ce6b5e1cf6760bd51ca87/mturk_annotation_viewer.py)

# 4. Yolo_mark
After receiving the annotated image data from Amazon Mechanical Turk, we decided we would generate the annotations ourselves manually. We utilized the Yolo_mark GUI created by AlexeyAB. In order to utilize this software, you will first need to download and install CMake, OpenCV, and Visual Studio. The download and installation instructions are below:
You will need to download the following programs in order to use Yolo mark, if you do not already have them:

#### OpenCV (Download as a ZIP)

https://github.com/opencv/opencv

#### CMake

https://cmake.org/download/

#### Microsoft Visual Studio (2017 version 15.9.15 garaunteed to work)

https://visualstudio.microsoft.com/vs/older-downloads/

### Installation (For Windows 10 64-bit)

1. Extract the opencv-master.zip to your C: or any drive you like, and you can rename the outfolder so the directory is like this `C:/OpenCV4/opencv-master`

2. Open the CMake GUI

3. For the "Where is the source code:" field, click **Browse Source...** and add the directory of the OpenCV folder from above. In this example it is `C:/OpenCV4/opencv-master`.

4. For the "Where to build the binaries:" field, create a folder inside the OpenCV folder called "opencv" and click **Browse Build...** to add it. In this example, it is `C:/OpenCV4/opencv4`

5. Click **Configure**
   - For "Specify the generator for this project", make sure to select your version of Visual Studio. In this example, we are using **Visual Studio 15 2017**
   - Under "Optional platform for generator", select **x64**. If you leave it blank, it defaults to Win32 and you will only be able to build Debug/Release in Visual Studio in Win32 and may get `LNK1112 module machine type 'x64' conflicts with target machine type 'x86` build error.
   - "Optional toolset to use" can  be left blank.
   - "Use default native compilers" can be checked.

6. After configuration is complete you will see a message `Configuring done` in the main GUI window. Find the field name **BUILD_opencv_world** in the main window and tick the box [x] to include it.

7. Click **Generate**
   - You will see a message `Generating done` in the main GUI window when it is complete.
   
8. Click **Open Project**
   - Visual Studio will open. You will need to find **CMakeTargets** folder in the Solution Explorer to the right. Make sure the Local Windows Debugger is set to **Debug** and **x64** and select the **ALL_BUILD** C++ project, right-click and select **BUILD**. 
     - If you happen to get an error:   
     `LNK1104: cannot open file 'python37_d.lib`
     - Then you can build the debug version of that library yourself if you do not have the file anywhere on your computer. It's very simple, please refer to this Stack Overflow post where **J.T. Davies** explains how to do it:
     https://stackoverflow.com/questions/17028576/using-python-3-3-in-c-python33-d-lib-not-found
   - After, set the Local Windows Debugger to **Release** and keep it set to **x64**. Select the **ALL_BUILD** C++ project again, right-click and select **BUILD**.
   - Keep the Local Windows Debugger on **Release** and set to **x64**. But this time, select the **INSTALL** C++ project, right-click and select **BUILD**.
   - Change the Local Windows Debugger to **Debug** and set to **x64**. Select the **INSTALL** C++ project, right-click and select **BUILD**.
     - This will generate all the library, includes, x64, and bin/lib directories all in one space.
   - You can close out of Visual Studio.
     
9. In Windows search, type `env` to access the Environment Variables.
   - In System Properties, under the Advanced tab, click **Environment Variables**
   - Below the **System variables** list, click **New** and enter these values for the two fields:
     - Variable name: `OPENCV_DIR`.
     - Variable value: `C:\OpenCV4\opencv4\install\x64\vc15`
   - Click OK
   - Still in the **System variables** window, select **Path** and click **Edit**. Add these two directories to the path:
     - %OPENCV_DIR%\bin
     - %OPENCV_DIR%\lib
   - Click OK and you can exit out of the System Properties now.
     
9. Test That Install Works
   - Creating an empty C++ project (Name it anything you want)
   - Right-click on the **Source Files** folder in the **Solution Explorer** to the right and add a **New Item** -> **C++ File (.cpp)** (sny name is fine)
   - Copy and paste the code below into the Source.cpp you just added:
     ```
     #include "opencv2/core.hpp"
     #include "opencv2/highgui.hpp"

        using namespace std;
        using namespace cv;

        int main(int argv, char* argc)
        {
	     Mat A;
	     A = Mat::zeros(100, 100, CV_8U);
	     namedWindow("x", WINDOW_AUTOSIZE);
	     imshow("x", A);
	     waitKey(0);
	     return 0;
        }	
	
   - Click on **View** -> **Other Windows** -> **Property Manager**.
   - A left sidebar will appear with **Debug | Win32**, **Debug | x64**, **Release | Win32**, and **Release | x64** options.
   - **Make sure the Local Windows Debugger is set to Debug and x64**
   - Right-click Debug | x64 in Property Manager window to the left and select **Properties**.
     - *Alternatively*, you can find these same C/C++, Linker options by right-clicking your C++ project in the Solution Explorer and selecting **Properties**.
   - Under **C/C++** -> **General** -> **Additional Include Directories**, add:
     - `C:\OpenCV4\opencv4\install\include`or whatever your install\include location is
   - Under **Linker** -> **Additional Library Dependences**, add:
     - `C:\OpenCV4\opencv4\install\x64\vc15\lib`
   - Under **Linker** -> **Input** -> **Additional Dependencies**, add:
     - `opencv_highgui411d.lib`
     - `opencv_core411d.lib`
   - Note: If your other projects are requiring other library files, you may need to add them to the **Linker** -> **Input** -> **Additional Dependencies** field manually. These files will have a "d" suffix for debug and are found in `C:\OpenCV4\opencv4\install\x64\vc15\lib`

Press **CTRL + F5** or the green play button next to Windows Local Debugger and a blank image should be created and be shown on the screen signifing the installation went successfully! :bowtie:
![Installation Complete!](https://github.com/alpizano/Yolo_mark/blob/master/testopencv.png)

# 5. Source code (Ultralytics)
This neural network, based on the YOLOv3 architecture makes use of 75 convlutional layers. This setup for this network can be found in the roulette3.cfg file:

https://github.com/alpizano/ultralytics/blob/beating-roulette/cfg/roulette3.cfg

For my GTX1060 (6GB) and for this 2 classese neural network:
- batches need to be set to 8
- subdivisions to 16
- max_batches to 400
- steps=3200,3600
- classes=2 
- filters=21

the roulette.names file has this structure:
```
classes= 2
train  = data/train.txt
valid  = data/test.txt
names = data/roulette.names
backup = backup/
```

#### To train
run 
```python train.py --data data/roulette.data --cfg cfg/roulette3.cfg --batch-size 8```
