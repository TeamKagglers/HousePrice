# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
# ===== Read Data ===== #
df0 = pd.read_csv("train.csv")
dfX = df0.drop(["SalePrice"], axis=1)
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("sample_submission.csv")
frames = [dfX, df1]
df = pd.concat(frames)

# ===== Find and add the columns having most NA values ===== #
null_df = dict(df.isnull().sum())
sort_null_df = dict(sorted(null_df.items(), key=lambda item: item[1], reverse=True))
dropped_df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
df = dropped_df

# ===== Find and add the columns with very little variability ===== #
std_vals = df.std()
df.std() > 20
df = df.drop(columns=['OverallQual', 'OverallCond', 'OverallCond', 'BsmtHalfBath', 'FullBath',
                      'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                      'GarageCars', 'MoSold', 'YrSold', 'Id'])

# ===== Find and add highly correlated columns which may produce multicollinearity problem ===== #

# YearBuilt - GarageYrBlt  [corr ~ 0.825667]  |  TotalBsmtSF - 1stFlrSF   [corr ~ 0.81953 ] #
# GrLivArea - TotRmsAbvGrd [corr ~ 0.825489]  |  GarageCars - GarageArea  [corr ~ 0.882475] #

plt.figure(figsize=(16, 6))
corr         = df.corr()
kot          = corr[corr >= 0.75]
sns.heatmap(kot, cmap=None)
plt.show()

df = df.drop(['GarageYrBlt', 'TotalBsmtSF', '1stFlrSF', 'GarageArea', 'YearBuilt'], axis=1)

# ===== Find and add columns having NA values ===== #

noncateg_columns_df = [c for c in list(df.select_dtypes(include=['int64', 'float64']).columns)]
noncateg_columns_with_null_df = [c for c in noncateg_columns_df if df[c].isnull().values.any()]
categ_columns_df = [c for c in list(df.select_dtypes(include=['object', 'bool']).columns)]
categ_columns_with_null_df = [c for c in categ_columns_df if df[c].isnull().values.any()]

# ===== Impute NA values ===== #
categ_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
noncateg_imp = SimpleImputer(missing_values=np.nan, strategy='mean')

df[categ_columns_with_null_df] = pd.DataFrame(categ_imp.fit_transform(df[categ_columns_with_null_df]))
df[noncateg_columns_with_null_df] = pd.DataFrame(noncateg_imp.fit_transform(df[noncateg_columns_with_null_df]))

# TODO: implement

# ===== One-Hot-Encode categorical columns [train && test data] ===== #
ohe = sklearn.compose.ColumnTransformer([
    ('1hot', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'),
     categ_columns_df),
])

ohe_df = pd.DataFrame(ohe.fit_transform(df))

# Renaming features names #

ohe_df.columns = ohe.get_feature_names()

# ===== Divide into Test and Train data (80% to 20%) ===== #


Y = ohe_df

Y_train = ohe_df.iloc[:2335]
Y_test = ohe_df.iloc[2335:]

X_train = pd.DataFrame(df0['SalePrice'])
X_test = pd.DataFrame(df2['SalePrice'])

X = pd.concat([X_train, X_test])

X_train=X.iloc[:2335, :]       #SalePrice
X_test=X.iloc[2335:, :]        #Saleprice

# For seeing SalePrice's distribution

plt.hist(X['SalePrice'])
plt.show

# Normalization of data #

X_train = np.log(X_train+1)
X_test = np.log(X_test+1)

# Changing DataFrame to Array for fitting #

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# Model Training #

lambdas = np.geomspace(0.01, 100000, num=100)
best_rmse = 99999
rmses = []

for l in lambdas:
    clf = Ridge(alpha=round(l, 2), tol=0.001, solver='auto')
    clf.fit(Y_train, X_train)
    predicted_X = clf.predict(Y_test)
    rmse = np.math.sqrt(sklearn.metrics.mean_squared_error(predicted_X, X_test))
    if rmse < best_rmse:
        best_lambda = round(l, 2)
        best_rmse = rmse
    print("Lambda: {:.2f} -- RMSE: {:.2f}".format(l, rmse))

print(f"Best RMSE: {best_rmse}")


