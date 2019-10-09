# -*- coding: utf-8 -*- 
import cv2
import sys
from PIL import Image
def catchUsbVedio(window_name,camera_index):
    cv2.namedWindow(window_name)
    
    cap=cv2.VideoCapture(camera_index)#视频来源，可以是视频文件，也可以是摄像头
    classfier=cv2.CascadeClassifier("D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
    color=(0,255,0)
    while cap.isOpened():
        ok,frame=cap.read()#读取一帧图片
        if not ok:
            break
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#获取灰度图片
        
        faceRects=classfier.detectMultiScale(grey,scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects)>0:
            for faceRect in faceRects:
                x,y,w,h=faceRect
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        
        cv2.imshow(window_name,frame)#显示图片并等待10ms，输入q退出
        c=cv2.waitKey(100)
        if c &0xFF ==ord("q"):
            break
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    
    catchUsbVedio("face",0)
