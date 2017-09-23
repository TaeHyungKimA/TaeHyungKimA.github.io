---
layout: post
title: Linear Regression 
---

**이번시간에는 Linear Regression대해 다뤄볼 예정입니다.**

Regression : x(data)-> y
Classification : x -> y(가나올확률)




Linear Regression Model이란 ?

->선형회귀라고 표현하며, 선형적인 데이터를 분석하고 예측하기 위한 모델입니다




![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear1.jpg)
\n
    [Linear Data]

```
# Linear Regression with Gradient Descent
import matplotlib.pyplot as plt
import numpy as np
import random
import math
 
 
a_true = random.randint(1, 10)
b_true = random.randint(1, 10)
 
#x = [float(x)+np.random.normal() for x in range(-20, 20, 1)]
x = [random.uniform(-10, 10) for x in range(60)]
y = [a_true * xi+b_true for xi in x]
print(x)
print(y)
 
 
def sigmoid(z):
    return 1 / (1 + math.e ** -z)
 
 
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
            gradient_b += 1
        elif ek < 0:
            gradient_a += (X[k])
            gradient_b += (-1)
        elif ek == 0:
            gradient_a += 0
            gradient_b += 0
            
    return loss, gradient_a, gradient_b


```