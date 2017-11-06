import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data= pd.read_csv('X:\Projetcs\Employee Predicition\HR_comma_sep.csv')
#print(data.head)
print(list(data.columns))
print(data.isnull().sum())

sns.countplot(y= data.sales, data=data, palette='hls')
plt.show()


    print("Yes")
