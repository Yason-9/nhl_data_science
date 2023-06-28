from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn 

digits = load_digits()

print(digits.data[0])
#generate an 8x8 vector, resmebling that of a number

print(digits.data)
print(len(digits.data))
#contains a dataset of a lot of vectors
print(digits.target)
print(len(digits.target))
#contains the dataset of the correct answers to digits.dataZ

#plt.gray()
#for i in range(5):
#    plt.matshow(digits.images[i])
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)

model = LogisticRegression()

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

plt.matshow(digits.images[67])
digits.target[67]

model.predict([digits.data[67]])
#need to provide a 2D array 
#gives the prediction based on the trained model 

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()