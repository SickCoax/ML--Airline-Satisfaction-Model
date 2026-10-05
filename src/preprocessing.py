import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def get_X_y(df) :

    ranges = [0 , 10 , 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100 , 200 , 300 , 400 , 500]

    j = 1 

    for i in range(len(ranges) - 1):

        condition1 = (
        (df["Departure Delay in Minutes"] > ranges[i]) & 
        (df["Departure Delay in Minutes"] <= ranges[j]) & 
        (df["Arrival Delay in Minutes"].isnull())
        )
        
        condition2 = (
        (df["Departure Delay in Minutes"] > ranges[i]) & 
        (df["Departure Delay in Minutes"] <= ranges[j])
        )
        
        col = "Arrival Delay in Minutes"

        fill = float((df["Departure Delay in Minutes"][condition2]).mean())

        fill = round(fill , 2)
        
        df.loc[condition1 , col] = fill
        
        j = j + 1

    df.loc[df["Arrival Delay in Minutes"].isnull() , "Arrival Delay in Minutes"] = 0

    X = df.drop(["satisfaction"] , axis = 1)
    y = df[["satisfaction"]]

    y = OrdinalEncoder().fit_transform(y)
    y = (pd.Series(y.flatten())).astype(int)

    return X , y


def get_X(df) :

    ranges = [0 , 10 , 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100 , 200 , 300 , 400 , 500]

    j = 1 

    for i in range(len(ranges) - 1):

        condition1 = (
        (df["Departure Delay in Minutes"] > ranges[i]) & 
        (df["Departure Delay in Minutes"] <= ranges[j]) & 
        (df["Arrival Delay in Minutes"].isnull())
        )
        
        condition2 = (
        (df["Departure Delay in Minutes"] > ranges[i]) & 
        (df["Departure Delay in Minutes"] <= ranges[j])
        )
        
        col = "Arrival Delay in Minutes"

        fill = float((df["Departure Delay in Minutes"][condition2]).mean())

        fill = round(fill , 2)
        
        df.loc[condition1 , col] = fill
        
        j = j + 1

    df.loc[df["Arrival Delay in Minutes"].isnull() , "Arrival Delay in Minutes"] = 0

    return df
