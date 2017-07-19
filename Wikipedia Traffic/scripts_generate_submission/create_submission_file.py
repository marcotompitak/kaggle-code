import pandas as pd
import numpy as np

# Load sample submission and our predictions
subm = pd.read_csv('../input/key_1.csv')
pred = pd.read_csv('../output/full_dataset_pred_median_weekday_split.dat')

# Merge the two
m = subm.merge(pred, how = 'left')

# Turns out some datasets contained only NaNs, and our
# forward and backward fillna procedures didn't work.
# Will be fixed in the next version of the analysis
# scripts, but for now a dirty fix setting everything
# to 0.
m['traffic'] = m['traffic'].fillna(0).astype(int)

# Fix header
m = m.rename(columns = {'traffic': 'Visits'})

# Generate submission file
m[['Id', 'Visits']].to_csv('../submission.csv', index = False)
