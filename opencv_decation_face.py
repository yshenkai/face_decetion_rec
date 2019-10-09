import cv2
import sys
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import numpy as np

model=load_model("face_model.h5")
def encoding_by_image_path(image_path,model):
    img=cv2.imread(image_path,1)
    img=img[:,:,(2,1,0)]
    img=np.around(img/255.,decimals=12)
    img=np.array(img)
    img=np.expand_dims(img,0)
    encoding=model.predict(img)
    return encoding

database={}
database["danielle"] = encoding_by_image_path("images/danielle.png", model)
database["younes"] = encoding_by_image_path("images/younes.jpg", model)
database["tian"] = encoding_by_image_path("images/tian.jpg", model)
database["andrew"] = encoding_by_image_path("images/andrew.jpg", model)
database["kian"] = encoding_by_image_path("images/kian.jpg", model)
database["dan"] = encoding_by_image_path("images/dan.jpg", model)
database["sebastiano"] = encoding_by_image_path("images/sebastiano.jpg", model)
database["bertrand"] = encoding_by_image_path("images/bertrand.jpg", model)
database["kevin"] = encoding_by_image_path("images/kevin.jpg", model)
database["felix"] = encoding_by_image_path("images/felix.jpg", model)
database["benoit"] = encoding_by_image_path("images/benoit.jpg", model)
database["arnaud"] = encoding_by_image_path("images/arnaud.jpg", model)
database["shenkai"]=encoding_by_image_path("images/shenkai.jpg",model)
#print(database)

def encoding_by_image(image,model):
    image=image[:,:,(2,1,0)]
    image=np.around(image/255.,decimals=12)
    image=np.array(image)
    image=np.expand_dims(image,0)
    encoding=model.predict(image)
    return encoding

def face_re(image,model,database):
    out_ecoding=encoding_by_image(image,model)
    
    min_cost=100
    
    for (name,encoding) in database.items():
        cost=np.linalg.norm(out_ecoding-encoding)
        if cost<min_cost:
            min_cost=cost
            identity=name
    if min_cost>0.7:
        return "unknow"
    else:
        return identity





def deteced_face_from_vedio(window_name,camera_index):
    cv2.namedWindow(window_name)
    vedio=cv2.VideoCapture(camera_index)
    classesfile=cv2.CascadeClassifier("D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
    color=(255,0,0)
    while vedio.isOpened:
        ok,image=vedio.read()#读取一帧图片
        if not ok:
            break
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        detects=classesfile.detectMultiScale(image_grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(detects)>0:
            for dectect in detects:
                x,y,w,h=dectect
                jq_image=image[y-10:y+h+10,x-10:x+w+10]
                cv2.rectangle(image,(x-10,y-10),(x+w+10,y+h+10),color,2)
                
                jq_image_reshape=cv2.resize(jq_image,(96,96))
                name=face_re(jq_image_reshape,model,database)
                #print(jq_image_reshape.shape)
                #cv2.imwrite("images/shenkai.jpg",jq_image_reshape)
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,name,(x+30,y+30),font,1,(255,0,255),4)
        cv2.imshow(window_name,image)
        cv2.waitKey(20)
    vedio.release()
    cv2.destroyAllWindows()



deteced_face_from_vedio("face_dection",0)
        
        
        