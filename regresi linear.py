import numpy as np
from sklearn.linear_model import LinearRegression

#Database
# x = Data, y = Target
x = [[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21]]
y = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63] #x dikali 3

regr = LinearRegression().fit(x,y)
regr.score(x,y)



#Data uji
predict = np.array([[4]]) #nilai yang di prediksi 

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
