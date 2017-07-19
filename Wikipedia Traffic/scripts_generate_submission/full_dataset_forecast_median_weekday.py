import pandas as pd
import numpy as np
from fbprophet import Prophet
import sys
import csv

"""
Calculate the median over the past so many days, but only
taking into account the weekdays.
"""
def past_median_weekday(data, days):
    past_data = data.iloc[-days:]
    return int(np.round(past_data[past_data['weekday']].median().values[0]))

"""
Calculate the median over the past so many days, but only
taking into account the weekends.
"""
def past_median_weekend(data, days):
    past_data = data.iloc[-days:]
    return int(np.round(past_data[~past_data['weekday']].median().values[0]))

"""
Generate a prediction with separate values for weekdays
and weekends.
"""
def predict_weekday_based(train, test, median_days):
    weekday_median = past_median_weekday(train, median_days)
    weekend_median = past_median_weekend(train, median_days)
    test_predict = test.copy()
    test_predict.loc[test_predict['weekday'].values, 'traffic'] = weekday_median
    test_predict.loc[~test_predict['weekday'].values, 'traffic'] = weekend_median
    test_predict['traffic'] = test_predict['traffic'].astype(int)
    return test_predict

# Command line arguments: filename to load and line to grab
data_filename = sys.argv[1]
data_line = int(sys.argv[2])
output_filename = sys.argv[3]

# Load in the training data
data = pd.read_csv(data_filename, header = 0, skiprows = range(1,data_line), nrows = 1, index_col = 0).T

# Grab the metadata
meta_string = data.columns.values[0]

# Rename the column to "traffic" instead of the metadata string
data.columns = ['traffic']

# Convert the index to a datetime
data.index = pd.to_datetime(data.index)

# Add a boolean it's-a-weekday column
data['weekday'] = data.index.weekday < 5

# Fill in missing values, using forward filling. However, often the
# first part of the dataset is missing, so we also apply backward filling.
data['traffic'] = data['traffic'].fillna(method='ffill').fillna(method='bfill')

# Generate dataframe with dates for which to predict
forecast = pd.DataFrame(index = pd.date_range('2017-01-01', '2017-03-01'))
forecast['weekday'] = forecast.index.weekday < 5

# Generate prediction
forecast = predict_weekday_based(data, forecast, 50)

# Convert date stamps to the format in key_1.csv, including the metadata
forecast['ds'] = [meta_string + "_" + x for x in forecast.index.strftime('%Y-%m-%d')]

# Drop weekday column
forecast = forecast.drop('weekday', axis=1)

# Output result
with open(output_filename, 'a') as predfile:
    forecast[['ds', 'traffic']].to_csv(predfile, index = False, header = False, quoting = csv.QUOTE_NONNUMERIC)

    
