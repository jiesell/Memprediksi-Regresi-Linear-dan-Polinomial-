from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

#Database
# x = Data, y = Target
x = [[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21]]
y = [1, 9, 25, 49, 81, 121, 169, 225, 289, 361, 441]  #x dipangkat 2

#Data uji
predict = np.array([[25]]) #nilai yang di prediksi 
poly = PolynomialFeatures(degree=2) #ordo yang digunakan
x_= poly.fit_transform(x) #fitting prediksi sumbu
predict = poly.fit_transform(predict) #fitting jenis regresi
regr = linear_model.LinearRegression()#meregresi
regr.fit(x_,y) #menentukan grafik


#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
