import pandas as pd
import numpy as np
from fbprophet import Prophet
import sys
import csv

# Calculate the median over the past so many days
def past_median(series, days):
    return series.iloc[-days:].median().values[0]

# Generate a prediction for a set of test dates
def predict(train, test, median_days):
    train_median = past_median(train, median_days)
    test_predict = test.copy()
    test_predict['traffic'] = train_median
    return test_predict

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

# Calculate SMAPE score
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

# Fill in missing values, using forward filling. However, often the
# first part of the dataset is missing, so we also apply backward filling.
data['traffic'] = data['traffic'].fillna(method='ffill').fillna(method='bfill')


# First we split into a training set and a validation set to get an idea
# of how well our forecasts work

ntrain = int((3/4)*len(data))
data_train = data.iloc[:ntrain]
data_test = data.iloc[ntrain:]

# Calculate score
score = SMAPE(predict(data_train, data_test, 50).values, data_test.values)


# Writing the scores to a file
with open(output_filename, "a") as scorefile:
    scorefile.write(str(score) + "\n")
