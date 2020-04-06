import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Verimizi csv dosyasından okuyoruz
#We're reading our data from csv file
data=pd.read_csv('Salary_Data.csv')


#Verilerimizdeki eksik verileri doldurabilmek için sklearn.preprocessing kütüphanesi altında Imputer'ı import ediyoruz. Burada amacımız verilerin ortalamasını eksik olan verilere yazdırmak. Bu eksik veriler verisetinde NaN olarak geçmektedir.
#In order to fill absent data in our data we import Imputer which is part of sklearn.preproccessing library. Our goal is here is writing the mean of all values into absent values. These absent values are represented by NaN in our dataser.
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy= 'mean',axis=0)

#verisetimizden experience_years kolonunu numpy array olarak seçiyoruz.
#we select experience_years column as numpy array
experience_years=data.iloc[:,0:1].values
experience_years[:,0:1]=imputer.fit_transform(experience_years[:,0:1])

#verisetimizden salary kolonunu numpy array olarak seçiyoruz.
#we select salary column as numpy array
salary=data.iloc[:,1:2].values
salary[:,0:1]=imputer.fit_transform(salary[:,0:1])

#Verisetimizi eğitim için test ve eğitim verisi olarak ikiye bölüyoruz. Bölerken yarı yarıya değil 0.33(test verisi boyutu) - 0.67(eğitim verisi boyutu) olarak bölüyoruz.
#We split our dataset to two for test and train data. While splitting we're not splitting 50-50, but splitting as 0.33(test data size) - 0.67(train data size)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(experience_years,salary,test_size=0.33,random_state=0)

#Böldüğümüz verileri aynı dünyaya indirmek için StandardScaler kullanarak Standardization yapılır. Örneğin experience_years 1-10 arasındayken salary ise 39343 ile 121872 arasındadır. 
#For reducing the data which we splitted into same world the Standardization operation is making via using StandarScaler. For example experience_years is between 1-10 while salary is between 39342-121872.
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
y_train=ss.fit_transform(y_train)
y_test=ss.fit_transform(y_test)

#Eğitim verilerimizi lineer regresyona sokup tahmin verilerimizi vererek bize bir tahmin gerçekleştirmesini istiyoruz.
#we gave our train data as input to linear regression and want it to predict corresponding values for our test data
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_hat_predict=lr.predict(x_test)

#verilerimizi x-eksenine göre sıralıyoruz. Ona uygun y-ekseni değerleri sıralanmış oluyor.
#we sort our values with respect to x-axis. The corresponding y-axis values are also be sorted.
x_train,y_train = zip(*sorted(zip(x_train, y_train)))

x_test,y_hat_predict=zip(*sorted(zip(x_test, y_hat_predict)))


#verilerimizi grafiğe döküyoruz.
#We plot our data to graph.
plt.scatter(x_test,y_hat_predict)
plt.plot(x_test,y_test,color="green")
plt.show()