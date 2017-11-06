import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.computation import align
from sklearn.preprocessing import StandardScaler

#from IPython import get_ipython
#get_ipython().magic('matplotlib inline')


df=pd.read_csv('HR_comma_sep.csv')
print(df.head())
print(df.shape)
print(df.corr())

correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, square=True, annot=True,cmap='cubehelix')
plt.title('Correlation between different fetaures')
plt.show()

df['sales'].unique()

sales = df.groupby('sales').sum()
print(sales)

groupby_sales = df.groupby('sales').mean()
print(groupby_sales)

IT=groupby_sales['satisfaction_level'].IT
RandD=groupby_sales['satisfaction_level'].RandD
accounting=groupby_sales['satisfaction_level'].accounting
hr=groupby_sales['satisfaction_level'].hr
management=groupby_sales['satisfaction_level'].management
marketing=groupby_sales['satisfaction_level'].marketing
product_mng=groupby_sales['satisfaction_level'].product_mng
sales=groupby_sales['satisfaction_level'].sales
support=groupby_sales['satisfaction_level'].support
technical=groupby_sales['satisfaction_level'].technical
print(technical)

department_name=('sales','IT','RandD','accoutning','hr','management','marketing','product_mng','support','technical')
department=(sales,accounting,product_mng,hr,technical,support,management,IT,RandD,marketing)
y_pos = np.arange(len(department))
x=np.arange(0,1,0.1)

plt.clf()
plt.barh(y_pos, department, align='center', alpha=0.8)
plt.yticks(y_pos, department_name)
plt.xlabel('Satisfaction levels')
plt.title('Mean satisfaction level of each department')
#plt.show()

df_drop=df.drop(labels = ['sales','salary'], axis=1)
print(df_drop.head())

cols=df_drop.columns.tolist()
cols.insert(0, cols.pop(cols.index('left')))
print(cols)

df_drop = df_drop.reindex(columns=cols)

X = df_drop.iloc[:,1:8].values
y = df_drop.iloc[:,0].values

print(X)
print(y)

X_std = StandardScaler().fit_transform(X)
cov_mat=np.cov(X_std.T)
print('Covariance matrix: \n%s' %cov_mat)

plt.clf()
sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between features')
#plt.show()

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda  x: x[0], reverse=True)

print('Eigenvalues in descending order')
for i in eig_pairs:
    print(i[0])

#print(eig_pairs[0][1])



tot=sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

plt.clf()

with plt.style.context(('dark_background')):
    plt.figure(figsize=(6,4))
    plt.bar(range(7), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

#plt.show()

plt.clf()

matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1),eig_pairs[1][1].reshape(7,1)))

print(matrix_w)

Y = X_std.dot(matrix_w)
print(Y)








