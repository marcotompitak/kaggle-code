Calculating SMAPE scores of a simple median model
-------------------------------------------------

These scripts test the simple past-median model on a data sample, split six-ways in order to parallellize the analysis on a six-core machine.

File descriptions:
 - `forecast_median.py`: Given a data file and the number of the line to read from it, this script splits the data into a training and test set, predicts the test values based on the median over the last 50 days in the training set, and calculates the [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) score of the prediction.
 - `analyze_median.sh`: Given a data file, it loops over its lines and feeds them to `forecast_median.py`.
 - `analyze_samples_parallel_median.sh`: Given a data set split into six (for a six-core machine) parts, run `analyze_median.sh` on each in parallel.
