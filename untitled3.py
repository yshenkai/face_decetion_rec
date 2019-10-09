
import cv2
import numpy as np

def face_decation(window_name,camera_index):
    cv2.namedWindow(window_name)
    vedio=cv2.VideoCapture(camera_index)
    classes=cv2.CascadeClassifier("D:\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml")
    color=(255,0,0)
    while vedio.isOpened:
        ok,frame=vedio.read()#读取状态
        if not ok:
            break
        
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detections=classes.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(detections)>0:
            for dect in detections:
                x,y,w,h=dect
                cv2.rectangle(frame,(x+10,y+10),(w+x+10,h+y+10),color,1)
                
            
        cv2.imshow(window_name,frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    vedio.release()
face_decation("face","te.mp4")
        