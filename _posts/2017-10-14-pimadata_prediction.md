---
layout: post
title: Deep Neural Network(Pima dataset)
---



<td>**이번시간에는 Pima dataset을 이용한 DNN(Deep Neural Network)에 대하여 알아볼 예정입니다.**</td>







++[Deep Neural Network]()란 ?++

Multilayer Perceptrons라고도 표현하며, 
여러개의 Hidden layer을 쌓아서 여러개의 가중치(weight) 합이 최종 Output결과로 나오게 된다.



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/nn1.jpg)

   
    
_ _ _

이러한 데이터들을 어떻게 잘 fit 시킬수 있을까?


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear2.JPG)

   
    
_ _ _

위 그림처럼 Data(x)가 있고,

정답 Label(y)이 있을경우 Data(x)로 부터 Label(y)을 예측하는것을

만드는것부터 출발합니다.


Model 설계 (hypothesis)

[y = ax + b  (a는 기울기 b는 bias)]

_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear3.JPG)
    
    
_ _ _
![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear4.JPG)


![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear5.JPG)


![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear6.JPG)


![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear7.JPG)

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear8.JPG)



지금까지 했던 내용을 코드화 시켜보면 아래와 같다.
```
# Linear Regression with Gradient Descent
import matplotlib.pyplot as plt
import numpy as np
import random
import math


def dataNormalization(x):
    x_average = 0 #Data 평균값
    x_var = 0     #Data 분산값
    x_delta = 0
    x_prime = []  #Data 정규화
    
    x_average = sum(x) / len(x)
    
    for k in x:
        x_delta = x_delta + (k - x_average)**2
    x_var = x_delta / len(x)
    #x_delta = x_delta / len(x)
    for k in range(len(x)):
        x_prime.append((x[k]-x_average)/(math.sqrt(x_var)))
    print("평균:",x_average,"분산:",x_var)
    return x_prime

def L2gradient(X, Y, a, b):
    loss = 0
    gradient_a = 0
    gradient_b = 0
    for k in range(len(X)):
        mk = X[k] * a + b
        ek = Y[k] - mk
        loss += ek ** 2
        gradient_a += ek * (-X[k])
        gradient_b += ek * (-1)
        
    return loss, gradient_a, gradient_b
 
def L1gradient(X, Y, a, b):
    loss = 0
    gradient_a = 0
    gradient_b = 0
    for k in range(len(X)):
        mk = X[k] * a + b
        ek = Y[k] - mk
        loss += abs(ek)
        if ek > 0:
            gradient_a += (-X[k])
            gradient_b += (-1)
        elif ek < 0:
            gradient_a += (X[k])
            gradient_b += 1
        elif ek == 0:
            gradient_a += 0
            gradient_b += 0
            
    return loss, gradient_a, gradient_b
 
 
a_true = random.randint(1, 10)
b_true = random.randint(1, 10)


#x = [float(x)+np.random.normal() for x in range(-20, 20, 1)] #가우시안 노이즈 추가
x = [random.randint(0, 10) for x in range(1000)]
x = dataNormalization(x) # data normalization
y = [a_true * xi+b_true for xi in x]
print(x)
print(y)
 

#fitting ->초기값도 잘바꿔서 테스트해볼수 있다
a_hat = 1.
b_hat = 1.
learning_rate = 1e-4
loss_graph = []
#update a_hat,b_hat
for iter in range(10000):
    loss,gradient_a,gradient_b = L1gradient(x, y, a_hat, b_hat) #L2gradient로 바꿔서 테스트
    loss_graph.append(loss)
    print("iter %d : loss = %f, a=%f b=%f" % (iter,loss,a_hat,b_hat))
    a_hat = a_hat - gradient_a * learning_rate
    b_hat = b_hat - gradient_b * learning_rate
 

 
print("iter final : loss = %e, a=%f b=%f" % (loss,a_hat,b_hat))
plt.plot(loss_graph)
plt.show()

y_ = [X*a_hat + b_hat for X in x]
plt.scatter(x,y)
plt.plot(x,y, 'r',label='a=true,b=true')
plt.plot(x,y_, 'y', label='a=a_hat,b=b_hat')
plt.legend()
plt.show()



```
