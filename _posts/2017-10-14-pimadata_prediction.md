---
layout: post
title: Deep Neural Network(Pima dataset)
---



<td>**이번시간에는 Pima dataset을 이용한 DNN(Deep Neural Network)에 대하여 알아볼 예정입니다.**</td>






_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima1.JPG)



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima2.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima3.JPG)


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima4.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima5.JPG)




_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima6.JPG)




_ _ _




지금까지 했던 내용을 코드화 시켜보면 아래와 같다.
```
from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten


#fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#load pima indians dataset
dataset = pd.read_csv("diabetes.csv")
dataset = np.array(dataset)
X = dataset[:,0:8]
Y = dataset[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state = seed)



# create model
model = Sequential()
#Dense 입출력 관련 (출력개수,입력개수,입력형상,활성화함수)  init:초기화 함수 이름 weight가 없을 때 적용
model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(28, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
#early_stopping = EarlyStopping(patience = 20) #조기 종료 시키기
#model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=16, callbacks=[early_stopping])
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=50)



#model evaluate()
scores = model.evaluate(X_test, y_test)
print("\n",scores,"\n",model.metrics_names)#merics 쟤다, 여기에 포함되어있음 loss,acc
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#model predict ->하면 일종의 probablity확률이 나옴
y_out = model.predict(X_test)
for k in range(y_out.shape[0]):
    if y_out[k] > 0.5:
        y_out[k] = 1
    else:
        y_out[k] = 0

 
count = 0
for k in range(y_out.shape[0]):
    if (y_test[k]==1 and y_out[k] ==1) or (y_test[k] == 0 and y_out[k] ==0):
        count +=1
        
accuracy = (count/y_out.shape[0]) * 100
print("Keras가 구한 정확도 %.2f%%" % (scores[1]*100))
print("내가 구한 정확도:",accuracy)




```


_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/pima7.JPG)




_ _ _