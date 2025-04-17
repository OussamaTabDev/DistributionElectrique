import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
cols = ["Result" , "C" , "B" , "A" , "Ia" , "Ib" , "Ic" , "Va" , "Vb" , "Vc"]
df = pd.read_csv("classData.csv" )
# print(df.head())

# df["Result"]

# for label in cols[:-1]:
#     plt.hist(df[df["Result"] == 1][label] , color= 'blue' , label="Positive" , alpha = 0.6 , density = True )
#     plt.hist(df[df["Result"] == 0][label] , color= 'red' , label="Negative" , alpha = 0.6 , density = True )
#     plt.title(label)
#     plt.ylabel("P(X)")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()
# print(len(X))

train , valid , test = np.split(df.sample(frac = 1) , [int(0.6 * len(df)) , int(0.8 * len(df))])

print(len(train[train["Result" == 1]]))

# x = dataframe[dataframe.cols[:-1]].values
# y = dataframe[dataframe.cols[:-1]].values
 