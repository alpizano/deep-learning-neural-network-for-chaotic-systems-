import cv2
print(cv2.__version__)

# br-batch-1 every 320 frame, yields 713 images
# batch_1 every 400 frame, yields 593 images

# br-batch-0 [60FPS] GOPRO0000 - GOPR0006 = 192 imgs 12-2-19

# br-batch-1 [100FPS] GOPR0037 - GOPR0046 = 231 imgs 12-2-19
# br-batch-2 [100FPS] GOPR0317 - GOPR0327= 226 imgs 12-2-19
# br-batch-only-wheel [100FPS] GOPR0666 - GOPR0667 = 241 imgs 12-2-19

#for i in range(37,139):
#for i in range(37,47):
#for i in range(0,7):
#for i in range(317,327):
for i in range(666,668):
    if i < 10:
        file_num = "00" + str(i)
    elif 10 <= i < 100:
        file_num = "0" + str(i)
    else:
        file_num = str(i)
    vidcap = cv2.VideoCapture('br-batch-3-only-wheel/GOPR0%s.MP4' % file_num)
    #vidcap = cv2.VideoCapture('GoPro_Test/GOPR0%s.MP4' % file_num)
    success,image = vidcap.read()
    count = 0
    success = True

    while success:
        vidcap.set(1,count)
        success,image = vidcap.read()
        cv2.imwrite("br-batch-3-only-wheel-splice/%s_frame%d.jpg" % (file_num,count), image)     # save frame as JPEG file
        #count += 320
        #count += 400
        count += 100 # for 100 fps 960 camera settings
        #count += 60

        frame_count =  int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count is: %d" % frame_count)
