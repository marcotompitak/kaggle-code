Calculating SMAPE scores of a weekday/weekend-based median model
-------------------------------------------------

These scripts test the past-median model with separate predictions for weekdays and weekends on a data sample, split six-ways in order to parallellize the analysis on a six-core machine.

File descriptions:
 - `forecast_median_weekday.py`: Given a data file and the number of the line to read from it, this script splits the data into a training and test set, predicts the test values based on the weekday/weekend median over the last 50 days in the training set, and calculates the [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) score of the prediction.
 - `analyze_median_weekday.sh`: Given a data file, it loops over its lines and feeds them to `forecast_median_weekday.py`.
 - `analyze_samples_parallel_median_weekday.sh`: Given a data set split into six (for a six-core machine) parts, run `analyze_median_weekday.sh` on each in parallel.
