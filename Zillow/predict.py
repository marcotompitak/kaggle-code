import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Full set of properties
population = pd.read_csv('input/properties_2016.csv')

# Properties for which to predict in order to 
unknown = pd.read_csv('input/sample_submission.csv')

# For consistency with the population file
unknown['parcelid'] = unknown['ParcelId']
unknown.drop('ParcelId', axis=1)

# Some variables are categorical, and we will split them up using OneHot encoding
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

# Some variables take on very many categorical values. For the sake of this exercise, we'll drop them.
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

# The dates for which we need to predict; column names as given in sample submission file
date_cols = [
    '201610',
    '201611',
    '201612',
    '201710',
    '201711',
    '201712',
]

# The dates for which we need to predict in string form
date_vals = [
    '2016-10-01',
    '2016-11-01',
    '2016-12-01',
    '2017-10-01',
    '2017-11-01',
    '2017-12-01',
]

# Creating our full dataset
df_unknown = unknown.merge(population, how='left', on='parcelid')

# Dropping selected columns
df_unknown = df_unknown.drop(cols_to_drop+date_cols, axis=1)

# Re-encoding categorical variables
df_unknown_cat = pd.get_dummies(df_unknown, columns=categorical_cols)

# Load up the model we trained in Zillow.ipynb
model = xgb.Booster()
model.load_model("xgb_outliersremoved.model")

# Start a clean DataFrame to store the predictions
df_sub = pd.DataFrame(df_unknown['parcelid'])

# Loop over the six dates for which we need to predict
for i in range(6):

    # Add transactiondate
    df_unknown_cat['transactiondate'] = date_vals[i]

    # Transforming the transaction date into an ordinal variable: number of days since 01-01-2015
    df_unknown_cat['transactiondate_ordinal'] = pd.to_datetime(df_unknown_cat['transactiondate'],infer_datetime_format=True) - datetime.date(2015,1,1)
    df_unknown_cat['transactiondate_ordinal'] = df_unknown_cat['transactiondate_ordinal'].dt.days
    df_unknown_cat = df_unknown_cat.drop('transactiondate', axis=1)

    # Creating our variables and targets
    X = df_unknown_cat.drop(["parcelid"], axis=1)
    
    # Convert to XGBoost DMatrix
    Xmat = xgb.DMatrix(X)

    # Predict
    y = model.predict(Xmat)

    # Collect results
    df_sub[date_cols[i]] = y

# For consistency with the sample submission file
df_sub = df_sub.rename(columns = {'parcelid': 'ParcelId'})

# Save submission to file
df_sub.to_csv('pred.csv', ',', index=False, float_format='%.4f')
