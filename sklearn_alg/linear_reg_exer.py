import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

'''
determine which datasets have the greatest correlation with 
employee retention. Current model only uses integer values, figure out 
integer implmenetation later 
'''

def convert_int(dataset):
    return 

def determine_best(data, Y):
    columns = []
    accuracy = []

    num_of_params = data.shape[1]
    for i in range(num_of_params):
        col_name = data.columns[i]

        X_train, X_test, y_train, y_test = train_test_split(data[[col_name]], Y, test_size=0.9)
        model = LogisticRegression()

        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        columns.append(col_name)
        accuracy.append(acc)
    
#    data = dict(columns = columns, accuracy = accuracy)
#
#    print(pd.DataFrame(data = data))

    #quick sort implementation

    for i in range(1, len(accuracy)):
        key = accuracy[i]
        key2 = columns[i]
        #the value we're moving around 
        j = i-1
        
        while j >= 0 and key < accuracy[j]:
            #first call of accuracy[j] is the value before key
            #if it moves, it continues to be hte value before key since 
            #key (accuracy[j+1]) and j were subject to j-=1
            accuracy[j+1] = accuracy[j]
            columns[j+1] = columns[j]
            #changes j+1 value to j value 
            j -= 1
            #by mmaking j smaller, ensure the same value is being moved
            # if it is smaller than the previous value,

        accuracy[j+1] = key
        #wihtout this swap, the position of j+1 would be the same as j, since 
        #[j+1] = [j] only swapped j+1
        # notice how the value of key is not saved anywhere until the end 
        columns[j+1] = key2

    columns = columns[:-1]
    accuracy = accuracy[:-1] #remove left

    data = dict(columns = columns, accuracy = accuracy)

    df = pd.DataFrame(data = data)

    return columns, accuracy, df


if __name__ == '__main__':
    df = pd.read_csv("HR_comma_sep.csv")
    print(df.columns)
    adjust = pd.DataFrame(df.T).to_numpy()
    
    print(len(adjust))
    for i in range(len(adjust)):
        pass

    #columns, accuracy, corr = determine_best(df, df.left)
    #highest_cor_columns = columns[-3:]
    #subdf = df[highest_cor_columns]
#
    #X_train, X_test, y_train, y_test = train_test_split(subdf, df.left, test_size=0.9)
#
    #model = LogisticRegression()
    #model.fit(X_train, y_train)
    #
    #print(model.score(X_test, y_test))
    '''
    for integer integration:
    - need to save dictionaries of the numerical equivalents of certain columsn 
      within the main funciton
    - change the 2D array within the main function to convert strings to their integer equivalents 
    - store all numerical equivalents in one dictionary with the format
    key = integer_value
    value = list of corresponding string values from different columns
    - store all string values that originiate from certain columns with the format
    key = column_name
    value = list of all string values belonging to said column 
    - if want to predict using string values, create funciton using the dictionaries to give an 
      equivalent string value 
    '''