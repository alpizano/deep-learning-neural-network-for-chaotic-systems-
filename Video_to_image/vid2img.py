import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('../../data/european-roulette-220.mp4')
success,image = vidcap.read()
count = 0
success = True


while success:
    vidcap.set(1,count)
    success,image = vidcap.read()
    cv2.imwrite("images/frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1000

frame_count =  int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
