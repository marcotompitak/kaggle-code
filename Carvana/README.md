CNN solution for the Carvana Image Segmentation Kaggle Competition
------------------------------------------------------------------

Results of my exercise in building a convolutional neural network for the [Carvana Kaggle Competition](https://www.kaggle.com/c/carvana-image-masking-challenge).

File descriptions:

 - `Carvana.ipynb`: Main analysis notebook.
 - `cnn_pred.py`: Python script for applying the model to batches of data.
 - `model.h5`: Generated model.
 - `pred_batches.sh`: Loop over all test data in batches and apply `cnn_pred.py`.
 - `process_test_data.sh`: Apply preprocessing using ImageMagick.
 - `notebook_images`: Directory containing some externally generated images presented in `Carvana.ipynb`.
