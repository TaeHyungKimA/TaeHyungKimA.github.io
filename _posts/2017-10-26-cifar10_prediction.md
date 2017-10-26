---
layout: post
title: Convolution Neural Network (using CIFAR-10 data)
---



<td>**이번시간에는 CIFAR-10 Data을 이용한 CNN(Convolution Neural Network)에 대하여 알아볼 예정입니다.**</td>






_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN1.JPG)



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN2.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN3.JPG)


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN4.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN5.JPG)




_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/Cifar10_CNN6.JPG)




_ _ _




지금까지 했던 내용을 코드화 시켜보면 아래와 같다.
```
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

batch_size = 32
num_classes = 10
epochs = 20


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# One hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=batch_size, verbose=2)

scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#모델 시각
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax. set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


```



_ _ _