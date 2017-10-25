---
layout: post
title: Convolution Neural Network (using Fashion-MNIST data)
---



<td>**이번시간에는 Fasion-MNIST Data을 이용한 CNN(Convolution Neural Network)에 대하여 알아볼 예정입니다.**</td>






_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN1.JPG)



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN2.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN3.JPG)


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN4.JPG)




_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN5.JPG)




_ _ _


![Gatok Jekyll Theme]({{site.baseurl}}/./images/Fashion_CNN6.JPG)




_ _ _




지금까지 했던 내용을 코드화 시켜보면 아래와 같다.
```
import numpy as np
import mnist_reader
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
import time


seed = 7
np.random.seed(seed)
#시간측정
start_time = time.time()


#data load
x_train, y_train = mnist_reader.load_mnist('data/', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/', kind='t10k')
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define a simple CNN model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=12, batch_size=128, verbose=2)
# Final evaluation of the model
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

#걸린시간
print("--- %s seconds ---" %(time.time() - start_time))



```



_ _ _