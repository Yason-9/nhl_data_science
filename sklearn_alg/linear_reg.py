import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

bought = [0,0,1,0,1,1,0,1,1,1,
          0,0,0,0,1,1,1,1,0,0,
          0,0,1,1,1,1,0]
age = [22, 25,47	,52	,46	,56	,55	,
       60	,62	,61	,18	,28	,27	,29	,49	,
       55	,25	,58	,19	,18	,21	,26	,40	,45	,
       50	,54	,23	]

data = dict(age = age, bought = bought)

df = pd.DataFrame(data = data)

plt.scatter(df.age, df.bought, marker = "+", color = "red")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought, test_size=0.1)
'''
essentially splitting the data into two parts
one part for training, one part for testing
test_size indicates the portion of data out of 1 that is used for training
the rest is used for testing 
returns lists based on the number of parameters, indicating which datasets 
are used for training, and the ones used for testing
'''

model = LogisticRegression()
model.fit(X_train, y_train)
#trains the logistic regression with X_train an dy_train

print(model.predict(X_test))
print(model.score(X_test, y_test))
model.predict_proba(X_test)