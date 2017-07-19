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
    return past_data[past_data['weekday']].median().values[0]

"""
Calculate the median over the past so many days, but only
taking into account the weekends.
"""
def past_median_weekend(data, days):
    past_data = data.iloc[-days:]
    return past_data[~past_data['weekday']].median().values[0]

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
    return test_predict

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def SMAPE(truth, prediction):
    numerator = np.abs(truth - prediction)
    denominator = np.abs(truth) + np.abs(prediction)
    score_array = div0(numerator, denominator)
    return 200*np.mean(score_array)

# Command line arguments: filename to load and line to grab
data_filename = sys.argv[1]
data_line = int(sys.argv[2])
output_filename = sys.argv[3]

# Load in the training data
data = pd.read_csv(data_filename, header = 0, skiprows = range(1,data_line-1), nrows = 1, index_col = 0).T

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


# First we split into a training set and a validation set to get an idea
# of how well our forecasts work

ntrain = int((3/4)*len(data))
data_train = data.iloc[:ntrain]
data_test = data.iloc[ntrain:]

# Calculate score
score = SMAPE(predict_weekday_based(data_train, data_test, 50)['traffic'].values, data_test['traffic'].values)


# Writing the scores to a file
with open(output_filename, "a") as scorefile:
    scorefile.write(str(score) + "\n")
