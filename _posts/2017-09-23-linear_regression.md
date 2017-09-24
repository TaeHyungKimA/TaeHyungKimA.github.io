---
layout: post
title: Linear Regression 
---

<td>**이번시간에는 Linear Regression대해 다뤄볼 예정입니다.**</td>




Regression : x(data)-> y
Classification : x -> y(가나올확률)




++==Linear Regression Model이란 ?==++

선형회귀라고 표현하며, 선형적인 데이터를 분석하고 예측하기 위한 모델입니다



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear1.jpg)
    [Linear Data Example]
    
_ _ _

이러한 데이터들을 어떻게 잘 fit 시킬수 있을까?


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear2.jpg)
    [Linear Data fit]
    
_ _ _

위 그림처럼 Data(x)가 있고,

정답 Label(y)이 있을경우 Data(x)로 부터 Label(y)을 예측하는것을

만드는것부터 출발합니다.


Model 설계 (hypothesis)

[y = ax + b  (a는 기울기 b는 bias)]

_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear3.JPG)
    [Linear Data fit]
    
_ _ _
우리가 원하는건 True model처럼 fit을 시키는 것인데,

Hypothesis를 True model와 근사하게 만들기 위하여

cost(error) function을 정의한다

cost(L1) 
$$
\sqrt{3x-1}+(1+x)^2
$$

cost(L2) = 

우리의 최종목적은 cost가 0에 근사하는 a와 b값을 구하는 것이

Linear Regression의 학습의 목표이다.