import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

# Importing the competition data
known = pd.read_csv('input/train_2016_v2.csv')
population = pd.read_csv('input/properties_2016.csv')
unknown = pd.read_csv('input/sample_submission.csv')

# Categorical variables
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

# Variables to remove
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

# Generate training data
df_known = known.merge(population, how='left', on='parcelid').drop(cols_to_drop, axis=1)

# Get dummy variables for categorical columns
df_known_cat = pd.get_dummies(df_known, columns=categorical_cols)

# Convert transaction date to an ordinal
df_known_cat['transactiondate_ordinal'] = pd.to_datetime(df_known_cat['transactiondate'],infer_datetime_format=True) - datetime.date(2015,1,1)
df_known_cat['transactiondate_ordinal'] = df_known_cat['transactiondate_ordinal'].dt.days
df_known_cat = df_known_cat.drop('transactiondate', axis=1)

# Generate features and target 
X = df_known_cat.drop(["logerror", "parcelid"], axis=1)
y = df_known_cat["logerror"]

# Parameter values to search
param_grid = {
    'learning_rate': [0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1]
}

# Setting up the model
regressor = xgb.XGBRegressor(
    learning_rate = 0.01,
    n_estimators = 2000,
    silent = 0,
    objective = "reg:linear",
    gamma = 0
)

# Setting up the grid search
search = GridSearchCV(
    estimator = regressor, 
    param_grid = param_grid, 
    scoring = 'neg_mean_absolute_error', 
    cv = 5, 
    verbose = 0
)

# Perform the search
search.fit(X,y)

# Get the cross-validation results for the mean absolute error
means = search.cv_results_['mean_test_score']
means = np.negative(np.array(means))

# Save the results
np.savetxt('means_eta.csv', means, delimiter=',')

# Plot the results
_, ax = plt.subplots(1,1)

ax.plot(param_grid['learning_rate'], means)

ax.set_xlabel('eta', fontsize=16)
ax.set_ylabel('CV Average Score', fontsize=16)
ax.legend(loc="best", fontsize=15)
ax.grid('on')

plt.savefig('img/tmp_eta2.png')