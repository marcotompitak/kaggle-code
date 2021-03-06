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
    'max_depth': list(range(2,21,4)),
    'min_child_weight': [1,2,3,5,10,25,50,100]   
}

# Setting up the model
regressor = xgb.XGBRegressor(
    learning_rate = 0.01,
    n_estimators = 750,
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
means = np.negative(np.array(means).reshape(len(param_grid['min_child_weight']),len(param_grid['max_depth'])))

# Save the results
np.savetxt('means_maxdepth_minchildweight.csv', means, delimiter=',')

# Plot the results
_, ax = plt.subplots(1,1)

for idx, val in enumerate(param_grid['min_child_weight']):
    ax.plot(param_grid['max_depth'], means[idx,:], '-o', label= 'min_child_weight' + ': ' + str(val))

ax.set_xlabel('max_depth', fontsize=16)
ax.set_ylabel('CV Average Score', fontsize=16)
ax.legend(loc="best", fontsize=15)
ax.grid('on')

plt.savefig('img/tmp.png')