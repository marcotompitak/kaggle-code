import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

known = pd.read_csv('input/train_2016_v2.csv')
population = pd.read_csv('input/properties_2016.csv')
unknown = pd.read_csv('input/sample_submission.csv')

categorical_cols = [
    'airconditioningtypeid',
    'architecturalstyletypeid',
    'buildingclasstypeid',
    'decktypeid',
    'fips',
    'fireplaceflag',
    'hashottuborspa',
    'heatingorsystemtypeid',
    'storytypeid',
    'typeconstructiontypeid',
    'taxdelinquencyflag',
]

cols_to_drop = [
    'rawcensustractandblock',
    'censustractandblock',
    'propertycountylandusecode',
    'propertylandusetypeid',
    'propertyzoningdesc',
    'regionidcounty',
    'regionidcity',
    'regionidzip',
    'regionidneighborhood',
    'taxdelinquencyyear'
]

df_known = known.merge(population, how='left', on='parcelid').drop(cols_to_drop, axis=1)

df_known_cat = pd.get_dummies(df_known, columns=categorical_cols)
df_known_cat['transactiondate_ordinal'] = pd.to_datetime(df_known_cat['transactiondate'],infer_datetime_format=True) - datetime.date(2015,1,1)
df_known_cat['transactiondate_ordinal'] = df_known_cat['transactiondate_ordinal'].dt.days
df_known_cat = df_known_cat.drop('transactiondate', axis=1)

X = df_known_cat.drop(["logerror", "parcelid"], axis=1)
y = df_known_cat["logerror"]

param_grid = {
    'learning_rate': [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1]
}

regressor = xgb.XGBRegressor(
    learning_rate = 0.01,
    n_estimators = 2000,
    silent = 0,
    objective = "reg:linear",
    gamma = 0
)

search = GridSearchCV(
    estimator = regressor, 
    param_grid = param_grid, 
    scoring = 'neg_mean_absolute_error', 
    cv = 5, 
    verbose = 0
)

search.fit(X,y)

means = search.cv_results_['mean_test_score']
means = np.negative(np.array(means))

np.savetxt('means_eta.csv', means, delimiter=',')

_, ax = plt.subplots(1,1)

ax.plot(param_grid['learning_rate'], means)

ax.set_xlabel('eta', fontsize=16)
ax.set_ylabel('CV Average Score', fontsize=16)
ax.legend(loc="best", fontsize=15)
ax.grid('on')

plt.savefig('img/tmp_eta2.png')