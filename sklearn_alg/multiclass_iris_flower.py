from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn 
import pandas as pd

def train_model(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.8)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    print(model.score(X_test, y_test))
    return model, X_test

def prediction(train_model, target):
    numerical_pred = train_model.predict([target])
    if numerical_pred == 0:
        return "setosa"
    elif numerical_pred == 1:
        return "versicolour"
    else:
        return "virginica"

iris = load_iris()
print(iris.data[0])

iris_df = pd.DataFrame(data = iris['data'], columns = iris['feature_names'])

model, testing = train_model(iris)
print(prediction(model, testing[0]))
