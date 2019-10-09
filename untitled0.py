import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Input
from keras.utils import np_utils
alpha="abcdefghijklmnopqrstuvwxyz"
char_to_int=dict((c,i) for i,c in enumerate(alpha))
int_to_char=dict((i,c) for i,c in enumerate(alpha))

seq_length=3
dataX=[]
dataY=[]
for i in range(0,len(alpha)-seq_length,1):
    seq_in=alpha[i:i+seq_length]
    seq_out=alpha[i+seq_length]
    dataX.append([char_to_int[c] for c in seq_in])
    dataY.append(char_to_int[seq_out])
    #print(seq_in,"->",seq_out)
X=np.reshape(dataX,(len(dataX),1,seq_length))
print(X.shape)

X=X/float(len(alpha))
Y=np_utils.to_categorical(dataY)

model=Sequential()
model.add(LSTM(32,input_dim=seq_length,input_length=1))
model.add(Dense(Y.shape[1],activation="softmax"))
model.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer='adam')
model.fit(X,Y,epochs=500,batch_size=1,verbose=2)



loss,acc=model.evaluate(X,Y,verbose=0)
print("acc",acc)



for patten in dataX:
    x=np.reshape(patten,(1,1,len(patten)))
    x=x/float(len(alpha))
    prediction=model.predict(x)
    index=np.argmax(prediction)
    result=int_to_char[index]
    seq_in=[int_to_char[value] for value in patten]
    print(seq_in,"->",result)
