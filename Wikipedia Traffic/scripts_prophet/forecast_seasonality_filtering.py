import pandas as pd
import numpy as np
from fbprophet import Prophet
import sys
import csv

"""
Calculate the SMAPE score
"""
def SMAPE(truth, prediction):
    numerator = np.abs(truth - prediction)
    denominator = np.abs(truth) + np.abs(prediction)
    return 200*np.mean(numerator/denominator)

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

# Fill in missing values, using forward filling. However, often the
# first part of the dataset is missing, so we also apply backward filling.
data['traffic'] = data['traffic'].fillna(method='ffill').fillna(method='bfill')


# First we split into a training set and a validation set to get an idea
# of how well our forecasts work

ntrain = int((3/4)*len(data))
data_train = data.iloc[:ntrain]
data_test = data.iloc[ntrain:]

# Set up the dataframe in the format Prophet needs
prophet_train = pd.DataFrame(index = range(len(data_train)))
prophet_train['ds'] = data_train.index.copy()
prophet_train['y'] = data_train['traffic'].values

# Fit a model
model = Prophet(yearly_seasonality = True)
model.fit(prophet_train)

# Predict on the training set
future = model.make_future_dataframe(periods = len(data)-ntrain)
forecast = model.predict(future)

# Calculate the SMAPE score
unfiltered_score = SMAPE(forecast['yhat'].values[ntrain:], data['traffic'].values[ntrain:])


# We also build a model with a rolling mean applied to alleviate the problem
# of outliers
prophet_train_filtered = pd.DataFrame(index = range(len(data_train)))
prophet_train_filtered['ds'] = data_train.index.copy()
prophet_train_filtered['y'] = prophet_train['y'].rolling(window = 4, min_periods = 1, center = True).median()
model = Prophet(yearly_seasonality = True)
model.fit(prophet_train_filtered)
future = model.make_future_dataframe(periods = len(data)-ntrain)
forecast_filtered = model.predict(future)
filtered_score = SMAPE(forecast_filtered['yhat'].values[ntrain:], data['traffic'].values[ntrain:])


# And without the yearly seasonality
model = Prophet(yearly_seasonality = False)
model.fit(prophet_train)

# Predict on the training set
future = model.make_future_dataframe(periods = len(data)-ntrain)
forecast = model.predict(future)

# Calculate the SMAPE score
unfiltered_noseasonality_score = SMAPE(forecast['yhat'].values[ntrain:], data['traffic'].values[ntrain:])


# No seasonality but filtered
model = Prophet(yearly_seasonality = False)
model.fit(prophet_train_filtered)

# Predict on the training set
future = model.make_future_dataframe(periods = len(data)-ntrain)
forecast = model.predict(future)

# Calculate the SMAPE score
filtered_noseasonality_score = SMAPE(forecast['yhat'].values[ntrain:], data['traffic'].values[ntrain:])

# Gather scores and output
scores = [unfiltered_score, filtered_score, unfiltered_noseasonality_score, filtered_noseasonality_score]
min_score = min(scores)
min_index = scores.index(min(scores))

with open(output_filename, "a") as scorefile:
    scorefile.write(meta_string + "\t" + str(scores[0]) + "\t" + str(scores[1]) + "\t" + str(scores[2]) + "\t" + str(scores[3]) + "\t" + str(min_index) + "\n")

