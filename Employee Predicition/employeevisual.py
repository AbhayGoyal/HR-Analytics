import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib as plot



df=pd.DataFrame.from_csv("HR_comma_sep.csv")
print(df.head())

col_name =df.columns[0]
df=df.rename(columns = {col_name:'satisfaction'})

#print "<{}>".format(df.columns[1])

reg=LinearRegression()
#satisfaction=np.reshape(df.satisfaction, newshape=(np.array(df.satisfaction).shape[0], 1))
satsi=df.satisfaction.reshape(1, -1)
reg.fit(satsi, df.left)
pre=reg.predict(0.75)

print(pre*100)






