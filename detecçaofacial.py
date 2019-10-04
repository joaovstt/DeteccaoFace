import cv2
import numpy as np
from matplotlib import pyplot as plt

faceCascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture('video/video.mp4')
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
size = (
    int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output_face = cv2.VideoWriter('video/video2.mp4', codec, 23.0, size)

i=0
while True:
    ret, frame = video_capture.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
  
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = frame[y:y+h, x:x+w]
        i+=1
        cv2.imwrite("imagem/face{}.jpg".format(i), img)

        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([cinza],[0],None,[256],[0,256])

        plt.hist(hist)
        plt.savefig("imagem/histogramaframe{}.jpg".format(i))
           
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
