import cv2
print(cv2.__version__)

# br-batch-1 every 320 frame, yields 713 images
# batch_1 every 400 frame, yields 593 images

#for i in range(37,139):
for i in range(37,65):
#for i in range(0,16):
    if i < 10:
        file_num = "00" + str(i)
    elif 10 <= i < 100:
        file_num = "0" + str(i)
    else:
        file_num = str(i)
    vidcap = cv2.VideoCapture('br-batch-1/GOPR0%s.MP4' % file_num)
    #vidcap = cv2.VideoCapture('GoPro_Test/GOPR0%s.MP4' % file_num)
    success,image = vidcap.read()
    count = 0
    success = True

    while success:
        vidcap.set(1,count)
        success,image = vidcap.read()
        cv2.imwrite("br-batch-1-splice/%s_frame%d.jpg" % (file_num,count), image)     # save frame as JPEG file
        #count += 320
        #count += 400
        count += 100 # for 100 fps 960 camera settings
        #count += 50

        frame_count =  int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count is: %d" % frame_count)
