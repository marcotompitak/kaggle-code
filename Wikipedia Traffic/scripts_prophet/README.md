Calculating SMAPE scores of a Facebook Prophet model
-------------------------------------------------

These scripts test the Prophet model on a data sample, split six-ways in order to parallellize the analysis on a six-core machine.

File descriptions:
 - `forecast_seasonality_filtering.py`: Given a data file and the number of the line to read from it, this script splits the data into a training and test set, and fits four different models to the training set: 
  * With yearly seasonality (periodicity), without filtering
  * With yearly seasonality, with a rolling median filter
  * Without yearly seasonality, without filter
  * Without yearly seasonality, with filter
For each model, it forecasts and calculates the [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error) score on the test set.
 - `analyze_seasonality_filtering.sh`: Given a data file, it loops over its lines and feeds them to `forecast_seasonality_filtering.py`.
 - `analyze_samples_parallel_seasonality_filtering.sh`: Given a data set split into six (for a six-core machine) parts, run `analyze_seasonality_filtering.sh` on each in parallel.
