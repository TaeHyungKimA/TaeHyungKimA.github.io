---
layout: post
title: Linear Regression 
---

<td>**이번시간에는 Linear Regression대해 다뤄볼 예정입니다.**</td>




Regression : x(data)-> y
Classification : x -> y(가나올확률)




++[Linear Regression Model]()이란 ?++

선형회귀라고 표현하며, 선형적인 데이터를 분석하고 예측하기 위한 모델입니다



_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear1.JPG)

    [Linear Data Example]
    
_ _ _

이러한 데이터들을 어떻게 잘 fit 시킬수 있을까?


_ _ _

![Gatok Jekyll Theme]({{site.baseurl}}/./images/linear2.JPG)

    [Linear Data fit]
    
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
