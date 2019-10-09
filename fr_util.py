import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
from keras.models import Model
from keras.layers import Input,Conv2D,Dense,MaxPooling2D,AveragePooling2D,BatchNormalization,Activation,ZeroPadding2D,concatenate,Flatten,Lambda
import h5py
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
_Flaot="float32"
WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]
conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}


def load_weight():
    weight_path="./weights"
    filename=filter(lambda f:not f.startswith("."),os.listdir(weight_path))
    paths={}
    weight_dict={}
    for name in filename:
        paths[name.replace(".csv","")]=weight_path+"/"+name
    #print(paths)
    for name in WEIGHTS:
        if "conv" in name:
            conv_w=genfromtxt(paths[name+"_w"],delimiter=",",dtype=None)
            conv_b=genfromtxt(paths[name+"_b"],delimiter=",",dtype=None)
            conv_w=np.reshape(conv_w,conv_shape[name])
            conv_w=np.transpose(conv_w,[2,3,1,0])
            weight_dict[name]=[conv_w,conv_b]
        elif "bn" in name:
            bn_w=genfromtxt(paths[name+"_w"],delimiter=",",dtype=None)
            bn_b=genfromtxt(paths[name+"_b"],delimiter=",",dtype=None)
            bn_m=genfromtxt(paths[name+"_m"],delimiter=",",dtype=None)
            bn_v=genfromtxt(paths[name+"_v"],delimiter=",",dtype=None)
            weight_dict[name]=[bn_w,bn_b,bn_m,bn_v]
        elif "dense" in name:
            dense_w=genfromtxt(weight_path+"/dense_w.csv",delimiter=",",dtype=None)
            dense_b=genfromtxt(weight_path+"/dense_b.csv",delimiter=",",dtype=None)
            dense_w=np.reshape(dense_w,(128,736))
            dense_w=np.transpose(dense_w,[1,0])
            weight_dict[name]=[dense_w,dense_b]
    return weight_dict
def load_weight_to_FaceNet(FRModel):
    weights=WEIGHTS
    weight_dict=load_weight()
    for name in weights:
        if FRModel.get_layer(name) is not None:
            FRModel.get_layer(name).set_weights(weight_dict[name])
        
               
def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
def image_to_encoding(image_path,model):#
    img1=cv2.imread(image_path,1)
    img1=img1[:,:,(2,1,0)]
    img=np.around(img1/255.,decimals=12)
    x_train=np.array(img)
    x_train=np.expand_dims(x_train,0)
    #print(x_train.shape)
    encoding=model.predict(x_train)
    return encoding
#we will test the image_to_encoding
#image_to_encoding("images/camera_2.jpg")
def image_to_encoding_image(image,model):

    img = np.around(image / 255.,decimals=12)
    x_train = np.array(img)
    x_train = np.expand_dims(x_train, 0)
    # print(x_train.shape)
    encoding = model.predict(x_train)
    return encoding



def inception_block_1a(x):
    x_3x3=Conv2D(96,(1,1),name="inception_3a_3x3_conv1")(x)
    x_3x3=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_3x3_bn1")(x_3x3)
    x_3x3=Activation("relu")(x_3x3)
    x_3x3=ZeroPadding2D(padding=(1,1))(x_3x3)
    x_3x3=Conv2D(128,(3,3),name="inception_3a_3x3_conv2")(x_3x3)
    x_3x3=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_3x3_bn2")(x_3x3)
    x_3x3=Activation("relu")(x_3x3)
    
    x_5x5=Conv2D(16,(1,1),name="inception_3a_5x5_conv1")(x)
    x_5x5=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_5x5_bn1")(x_5x5)
    x_5x5=Activation("relu")(x_5x5)
    x_5x5=ZeroPadding2D(padding=(2,2))(x_5x5)
    x_5x5=Conv2D(32,(5,5),name="inception_3a_5x5_conv2")(x_5x5)
    x_5x5=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_5x5_bn2")(x_5x5)
    x_5x5=Activation("relu")(x_5x5)
    
    x_pool=MaxPooling2D(pool_size=3,strides=2)(x)
    x_pool=Conv2D(32,(1,1),name="inception_3a_pool_conv")(x_pool)
    x_pool=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_pool_bn")(x_pool)
    x_pool=Activation("relu")(x_pool)
    x_pool=ZeroPadding2D(padding=((3,4),(3,4)))(x_pool)
    
    x_1x1=Conv2D(64,(1,1),name="inception_3a_1x1_conv")(x)
    x_1x1=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3a_1x1_bn")(x_1x1)
    x_1x1=Activation("relu")(x_1x1)
    
    inception=concatenate([x_3x3,x_5x5,x_pool,x_1x1],axis=3)
    return inception

    
def inception_block_1b(x):
    x_3=Conv2D(96,(1,1),name="inception_3b_3x3_conv1")(x)
    x_3=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_3x3_bn1")(x_3)
    x_3=Activation("relu")(x_3)
    x_3=ZeroPadding2D(padding=(1,1))(x_3)
    x_3=Conv2D(128,(3,3),name="inception_3b_3x3_conv2")(x_3)
    x_3=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_3x3_bn2")(x_3)
    x_3=Activation("relu")(x_3)
    
    x5=Conv2D(32,(1,1),name="inception_3b_5x5_conv1")(x)
    x5=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_5x5_bn1")(x5)
    x5=Activation("relu")(x5)
    x5=ZeroPadding2D(padding=(2,2))(x5)
    x5=Conv2D(64,(5,5),name="inception_3b_5x5_conv2")(x5)
    x5=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_5x5_bn2")(x5)
    x5=Activation("relu")(x5)
    
    xpool=AveragePooling2D(pool_size=(3,3),strides=(3,3))(x)
    xpool=Conv2D(64,(1,1),name="inception_3b_pool_conv")(xpool)
    xpool=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_pool_bn")(xpool)
    xpool=Activation("relu")(xpool)
    xpool=ZeroPadding2D(padding=(4,4))(xpool)
    
    x1=Conv2D(64,(1,1),name="inception_3b_1x1_conv")(x)
    x1=BatchNormalization(axis=3,epsilon=0.00001,name="inception_3b_1x1_bn")(x1)
    x1=Activation("relu")(x1)
    inception=concatenate([x_3,x5,xpool,x1],axis=3)
    return inception

def conv2d_and_bn(x,layer=None,cv1_filter=(1,1),cv1_strides=(1,1),cv1_out=None,cv2_filter=(3,3),cv2_strides=(1,1),cv2_out=None,padding=None):
    if cv2_out is None:
        name=""
    else:
        name="1"
    x=Conv2D(cv1_out,cv1_filter,strides=cv1_strides,name=layer+"_conv"+name)(x)
    x=BatchNormalization(axis=3,epsilon=0.00001,name=layer+"_bn"+name)(x)
    x=Activation("relu")(x)
    if padding is None:
        return x
    x=ZeroPadding2D(padding=padding)(x)
    if cv2_out is None:
        return x
    x=Conv2D(cv2_out,cv2_filter,strides=cv2_strides,name=layer+"_conv2")(x)
    x=BatchNormalization(axis=3,epsilon=0.00001,name=layer+"_bn2")(x)
    x=Activation("relu")(x)
    return x 
def inception_block_1c(x):
    x3=conv2d_and_bn(x,layer="inception_3c_3x3",cv1_out=128,cv1_filter=(1,1),cv2_out=256,cv2_filter=(3,3),cv2_strides=(2,2),padding=(1,1))
    x5=conv2d_and_bn(x,layer="inception_3c_5x5",cv1_out=32,cv1_filter=(1,1),cv2_out=64,cv2_filter=(5,5),cv2_strides=(2,2),padding=(2,2))
    xpool=MaxPooling2D(pool_size=3,strides=2)(x)
    xpool=ZeroPadding2D(padding=((0,1),(0,1)))(xpool)
    inception=concatenate([x3,x5,xpool],axis=3)
    return inception
def inception_block_2a(x):
    x3=conv2d_and_bn(x,layer="inception_4a_3x3",cv1_filter=(1,1),cv1_out=96,cv2_out=192,cv2_filter=(3,3),cv2_strides=(1,1),padding=(1,1))
    x5=conv2d_and_bn(x,layer="inception_4a_5x5",cv1_out=32,cv1_filter=(1,1),cv2_out=64,cv2_filter=(5,5),cv2_strides=(1,1),padding=(2,2))
    xpool=AveragePooling2D(pool_size=(3,3),strides=(3,3))(x)
    xpool=conv2d_and_bn(xpool,layer="inception_4a_pool",cv1_out=128,cv1_filter=(1,1),padding=(2,2))
    x1=conv2d_and_bn(x,layer="inception_4a_1x1",cv1_out=256,cv1_filter=(1,1))
    inception=concatenate([x3,x5,xpool,x1],axis=3)
    return inception
def inception_block_2b(x):
    x3=conv2d_and_bn(x,layer="inception_4e_3x3",cv1_out=160,cv1_filter=(1,1),cv2_out=256,cv2_filter=(3,3),cv2_strides=(2,2),padding=(1,1))
    x5=conv2d_and_bn(x,layer="inception_4e_5x5",cv1_filter=(1,1),cv1_out=64,cv2_out=128,cv2_filter=(5,5),cv2_strides=(2,2),padding=(2,2))
    xpool=MaxPooling2D(pool_size=3,strides=2)(x)
    xpool=ZeroPadding2D(padding=((0,1),(0,1)))(xpool)
    inception=concatenate([x3,x5,xpool],axis=3)
    return inception
def inception_block_3a(x):
    x3=conv2d_and_bn(x,layer="inception_5a_3x3",cv1_out=96,cv1_filter=(1,1),cv2_out=384,cv2_filter=(3,3),cv2_strides=(1,1),padding=(1,1))
    xpool=AveragePooling2D(pool_size=(3,3),strides=(3,3))(x)
    xpool=conv2d_and_bn(xpool,layer="inception_5a_pool",cv1_out=96,cv1_filter=(1,1),padding=(1,1))
    x1=conv2d_and_bn(x,layer="inception_5a_1x1",cv1_out=256,cv1_filter=(1,1))
    inception=concatenate([x3,xpool,x1],axis=3)
    return inception
def inception_block_3b(x):
    x3=conv2d_and_bn(x,layer="inception_5b_3x3",cv1_out=96,cv1_filter=(1,1),cv2_out=384,cv2_filter=(3,3),cv2_strides=(1,1),padding=(1,1))
    xpool=MaxPooling2D(pool_size=3,strides=2)(x)
    xpool=conv2d_and_bn(xpool,layer="inception_5b_pool",cv1_out=96,cv1_filter=(1,1))
    xpool=ZeroPadding2D(padding=(1,1))(xpool)
    x1=conv2d_and_bn(x,layer="inception_5b_1x1",cv1_out=256,cv1_filter=(1,1))
    inception=concatenate([x3,xpool,x1],axis=3)
    return inception
def get_faceModel(input_shape):
    x_input=Input(shape=input_shape)
    x=ZeroPadding2D((3,3))(x_input)
    x=Conv2D(64,(7,7),strides=(2,2),name="conv1")(x)
    x=BatchNormalization(axis=3,name="bn1")(x)
    x=Activation("relu")(x)
    x=ZeroPadding2D((1,1))(x)
    x=MaxPooling2D(pool_size=(3,3),strides=2)(x)
    x=Conv2D(64,(1,1),strides=(1,1),name="conv2")(x)
    x=BatchNormalization(axis=3,epsilon=0.00001,name="bn2")(x)
    x=ZeroPadding2D((1,1))(x)
    x=Conv2D(192,(3,3),strides=(1,1),name="conv3")(x)
    x=BatchNormalization(axis=3,epsilon=0.00001,name="bn3")(x)
    x=Activation("relu")(x)
    x=ZeroPadding2D((1,1))(x)
    x=MaxPooling2D(pool_size=3,strides=2)(x)
        # Inception 1: a/b/c
    x = inception_block_1a(x)
    x = inception_block_1b(x)
    x = inception_block_1c(x)
    
    # Inception 2: a/b
    x = inception_block_2a(x)
    x = inception_block_2b(x)
    
    # Inception 3: a/b
    x = inception_block_3a(x)
    x = inception_block_3b(x)
    
    x=AveragePooling2D(pool_size=(3,3),strides=(1,1))(x)
    x=Flatten()(x)
    x=Dense(128,name="dense_layer")(x)
    x=Lambda(lambda x:K.l2_normalize(x,axis=1))(x)
    
    model=Model(inputs=x_input,outputs=x,name="faceModel")
    return model

frModel=get_faceModel(input_shape=(96,96,3))
#print("total param",frModel.count_params()) 
#print(frModel.summary())


#we will define the model's cost function to compile this model
#so we start do this job#三元损失函数
def compute_face_model_cost(y_true,y_pred,alpha=0.2):
    anchor,positive,negative=y_pred[0],y_pred[1],y_pred[2]
    post_dist=tf.reduce_sum(tf.square(anchor-positive))
    neg_dist=tf.reduce_sum(tf.square(anchor-negative))
    basic_loss=post_dist-neg_dist+alpha
    loss=tf.reduce_sum(tf.maximum(basic_loss,0.))
    
    return loss

#frModel.compile(loss=compute_face_model_cost,optimizer="adam",metrics=["accuracy"])
load_weight_to_FaceNet(frModel)

database={}
database["danielle"] = image_to_encoding("images/danielle.png", frModel)
database["younes"] = image_to_encoding("images/younes.jpg", frModel)
database["tian"] = image_to_encoding("images/tian.jpg", frModel)
database["andrew"] = image_to_encoding("images/andrew.jpg", frModel)
database["kian"] = image_to_encoding("images/kian.jpg", frModel)
database["dan"] = image_to_encoding("images/dan.jpg", frModel)
database["sebastiano"] = image_to_encoding("images/sebastiano.jpg", frModel)
database["bertrand"] = image_to_encoding("images/bertrand.jpg", frModel)
database["kevin"] = image_to_encoding("images/kevin.jpg", frModel)
database["felix"] = image_to_encoding("images/felix.jpg", frModel)
database["benoit"] = image_to_encoding("images/benoit.jpg", frModel)
database["arnaud"] = image_to_encoding("images/arnaud.jpg", frModel)



#下面定义一个验证函数
def verify(img_path,identity,database,model):
    out_encoding=image_to_encoding(img_path,model)
    #下面计算l2损失
    cost=np.linalg.norm(out_encoding-database[identity])
    print(cost)
    if cost<0.7:
        print("it is "+str(identity)+"!!!!!")
    else:
        print("it is not "+str(identity)+"!!!!")
        
#下面定义一个识别函数
def face_recognition(img_path,database,model):
    out_encoding=image_to_encoding(img_path,model)
    
    max_cost=100
    
    for (name,enc) in database.items():
        
        cur_cost=np.linalg.norm(out_encoding-database[name])
        if cur_cost<max_cost:
            max_cost=cur_cost
            identity=name
    if max_cost>0.7:
        return "unknow"
    else:
        return identity+" "+str(1-max_cost)
def face_recognition_image(image,model=frModel,database=database):
    out_encoding=image_to_encoding_image(image,model)
    
    max_cost=100
    
    for (name,enc) in database.items():
        
        cur_cost=np.linalg.norm(out_encoding-database[name])
        if cur_cost<max_cost:
            max_cost=cur_cost
            identity=name
    if max_cost>0.7:
        return "unknow"
    else:
        return identity+" "+str(1-max_cost)
##let me test this verify function 
#verify("images/camera_0.jpg", "kian", database,frModel)
frModel.save("face_model.h5")
#face_recognition("images/camera_3.jpg",database,frModel)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    