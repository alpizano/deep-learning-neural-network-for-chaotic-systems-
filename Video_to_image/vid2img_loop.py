import cv2
print(cv2.__version__)

for i in range(37,139):
    if i < 100:
        file_num = "0" + str(i)
    else:
        file_num = str(i)
    vidcap = cv2.VideoCapture('Beating_Roulette_Data/GOPR0%s.MP4' % file_num)
    success,image = vidcap.read()
    count = 0
    success = True

    while success:
        vidcap.set(1,count)
        success,image = vidcap.read()
        cv2.imwrite("images/%s_frame%d.jpg" % (file_num,count), image)     # save frame as JPEG file
        count += 32

        frame_count =  int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame count is: %d" % frame_count)
