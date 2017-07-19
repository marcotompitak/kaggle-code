Forecasting using a weekday/weekend-based median model
-------------------------------------------------

These scripts us the past-median model with separate predictions for weekdays and weekends on the full competition data, split six-ways in order to parallellize the analysis on a six-core machine.

File descriptions:
 - `full_dataset_forecast_median_weekday.py`: Given a data file and the number of the line to read from it, predicts the traffic in the period required in the competition based on the weekday/weekend median over the last 50 days in the data set.
 - `full_dataset_analyze_median_weekday.sh`: Given a data file, it loops over its lines and feeds them to `full_dataset_forecast_median_weekday.py`.
 - `full_dataset_analyze_samples_parallel_median_weekday.sh`: Given a data set split into six (for a six-core machine) parts, run `full_dataset_analyze_median_weekday.sh` on each in parallel.
 - `create_submission_file.py`: Take the generated predictions and format them into a submission file.
