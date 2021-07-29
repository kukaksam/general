import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier,RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
sns.set()

def linear(X,Y,X_test):
    reg = LinearRegression()
    reg.fit(X,Y)
    print(reg.score(X,Y))
    pred = reg.predict(X_test)
    return pred


df_train = pd.read_csv('train_houses.csv')
df_test = pd.read_csv('test_houses.csv')

#sns.distplot(df_train['SalePrice'])
#sns.distplot(np.log1p(df_train['SalePrice']))
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
df_train['GrLivArea'] = np.log1p(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log1p(df_test['GrLivArea'])


#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt']
#sns.pairplot(df_train[cols], size = 2)
#plt.show()


ds = pd.DataFrame({'Feature' : df_train.columns,'Missing':df_train.isnull().sum()})
ds = ds[ds['Missing']>0]
cols = ds['Feature']



#print(ds[ds['Missing']>0].sort_values(ascending=False,by='Missing'))
df_train = df_train.drop(cols,axis=1)
df_train = pd.get_dummies(df_train)
df_test = df_test.drop(cols,axis=1)


df_test = pd.get_dummies(df_test)
Y_train = df_train['SalePrice']
X_train = df_train.drop('SalePrice',axis=1)

dd = pd.DataFrame({'Feature' : df_test.columns,'Missing':df_test.isnull().sum()})
d = dd[dd['Missing']>0]
misscols = d['Feature']
for item in misscols:
    df_test[item] = df_test.fillna(df_test[item].median())

diff = list(set(X_train.columns)-set(df_test.columns))
print(diff)
#df_test = df_test.drop(diff,axis=1)
X_train = X_train.drop(diff,axis=1)
print(diff)
print(len(X_train.columns),len(df_test.columns))




pred = linear(X_train,Y_train,df_test)


df = pd.DataFrame({'Id':df_test['Id'],'SalePrice':pred})
df = df.set_index('Id')
df.to_csv('iowa.csv')

