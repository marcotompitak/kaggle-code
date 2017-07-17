XGBoost model for the Zillow Kaggle Competition
-----------------------------------------------

Results of my exercise to learn XGBoost by applying it to the data from the [Zillow Kaggle Competition](https://www.kaggle.com/c/zillow-prize-1).

File descriptions:

 - `Zillow.ipynb`: Jupyter notebook containing my main analysis and model construction.
 - `Zillow.html`: Static html version of the notebook.
 - `CV_eta.py`: Uses scikit-learn's GridSearchCV to find an optimal value for the learning rate.
 - `CV_maxdepth_minchildweight.py`: GridSearchCV for optimal `max_depth` and `min_child_weight` parameters.
 - `means_eta.csv` and `means_maxdepth_minchildweight.csv`: Results of the grid searches.
 - `predict.py`: Python script to generate a submission file for the competition.
 - `img`: Directory containing visualizations of the grid search results for use in the notebook.
