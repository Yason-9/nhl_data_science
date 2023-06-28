import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import preprocessing

def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    # X contains the paramter values of the dataset
    # y_true contains the actual/desired values   
    number_of_features = X.shape[1]
    # X is a 31x3 array 
    print(X.shape[0])
    print(number_of_features)
    print(X)
    # gives the number of parameters for the model

    #.shape gives the dimensions of the array 
    #.shape[0] gives the number of rows
    #.shape[1] gives the numebr of columsn

    w = np.ones(shape=(number_of_features))
    print(w)
    # creates a 1xn array with ones 
    b = 0
    # bias value, the "y-intercept"
    total_samples = X.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted =  np.dot(w, X.T) + b
        
        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted))
        #weight derivative calculations
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        #bias adjustement( don't need the extra X multiplication)
        #derivative calculations, for each training point in the set 

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        #adjusting values after training sets d

        cost = np.mean(np.square(y_true-y_predicted))
        #numpy array, generating the RMSE 

        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    
    return w, b, cost, cost_list, epoch_list

def prediction(g_for, g_against, pp_p, w, b):
    #keep the scaling functions the same between functions 

    scaled_X = sx.transform([[g_for, g_against, pp_p]])[0]
    #need to give 0th index, requires 2-dimensional array
    scaled_wr = w[0]*scaled_X[0] + w[1] * scaled_X[1] + w[2] * scaled_X[2] + b
    return sy.inverse_transform([[scaled_wr]])[0][0]
    # obtaining the predicted win rate based off the generated values 
    # use [0][0] to search 0th value of the 0th array

win_percentage = [75.61 ,60.98 ,59.76 ,58.54 ,58.54 ,57.31 ,57.31 ,57.31 ,56.10 ,56.10 ,56.10,
                  54.88,53.66 ,53.66 ,52.44 ,52.44 ,47.56 ,46.34 ,
                  45.12 ,45.12 ,43.90 ,43.90 ,42.68 ,42.68 ,42.68 ,
                  40.24 ,39.02 ,39.02 ,37.80 ,37.80 ,35.37]

goals_for = [3.89, 3.52,3.13 ,2.72 ,3.34, 2.88,3.12,3.29,3.49,2.96,
             3.52,2.98,3.30,3.00,2.55,3.00,2.55,3.15,2.94,2.56,3.22,
             3.26,2.79,2.67,2.39,2.70,2.70,2.73,2.67,2.43,2.95]

goals_against = [2.70,2.72,2.59,2.33,3.02,2.59,2.82,2.96,3.04,
                 2.70,3.15,2.68,2.90,2.88,2.44,2.78,2.68,2.98,
                 3.41,2.84,3.33,3.55,3.30,3.02,3.02,3.27,3.26,
                 3.32,3.30,3.6,3.67]

pp_percentage = [28.2,19.3,25.9,14.5,20.8,12.9,15.4,24.8,21.8,17.8,23.7,
                 21.1,24.6,13.3,21.0,16.8,16.3,22.0,17.1,20.3,26.8,20.2,
                 21.2,17.1,17.0,19.5,19.4,18.1,17.7,15.8,20.4]


data = dict(win_percentage = win_percentage, goals_for = goals_for,
            goals_against = goals_against, pp_percentage = pp_percentage)
df = pd.DataFrame(data=data)

diff = df.drop("win_percentage", axis = "columns")
#removes the win% column 

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

scaled_X = sx.fit_transform(df.drop('win_percentage',axis='columns'))
#rescales the values in a range from 0 to 1
scaled_Y = sy.fit_transform(df['win_percentage'].values.reshape(df.shape[0],1))

kekw = scaled_Y.reshape(scaled_Y.shape[0],)
#transposes the matrix
'''
Machine Learning Model
h(x) = Ax + Bx + Cx + D
s.t A, B, C are parameters and D is the bias 
'''
asdf = np.ones(shape=10)
#print(asdf)
#print(scaled_X)

w, b, cost, cost_list, epoch_list = batch_gradient_descent(
    scaled_X,
    scaled_Y.reshape(scaled_Y.shape[0],),
    #transposes the matrix 
    1000
)

win = (prediction(2.94, 3.41, 17.1, w, b))

print(win)

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list, cost_list)
plt.show()