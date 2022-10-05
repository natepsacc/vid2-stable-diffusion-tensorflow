import cv2
from PIL import Image



class createframes:
  vidcap = cv2.VideoCapture('4.mp4')
  success,image = vidcap.read()
  count = 0


  while success:
    cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file  



    


    success,image = vidcap.read()
    print('Read a new frame: %d ' % (count +1), success)

    count += 1
