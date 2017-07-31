from keras.models import load_model
from keras.layers import Conv2D
from keras.models import Sequential
import keras.backend as K
import pandas as pd
import numpy as np
import PIL.Image as Image
from scipy.ndimage import binary_fill_holes
import sys

## Helper functions, see notebook

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    print(intersection)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def cut_relevant(rgba_img):
    nonempty_cols = np.where(rgba_img[:,:,3].max(axis=0)>0)[0]
    nonempty_rows = np.where(rgba_img[:,:,3].max(axis=1)>0)[0]
    cropbox = (
        min(nonempty_rows), 
        max(nonempty_rows), 
        min(nonempty_cols), 
        max(nonempty_cols)
    )
    new_img = rgba_img[cropbox[0]:cropbox[1]+1, cropbox[2]:cropbox[3]+1, :]
    return new_img

def get_relevant_box(rgba_img):
    nonempty_cols = np.where(rgba_img[:,:,3].max(axis=0)>0)[0]
    nonempty_rows = np.where(rgba_img[:,:,3].max(axis=1)>0)[0]
    cropbox = (
        min(nonempty_rows), 
        max(nonempty_rows), 
        min(nonempty_cols), 
        max(nonempty_cols)
    )
    return cropbox

def fix_size(rgba_img):
    rgb_img = np.array(
        Image.fromarray(
            cut_relevant(
                rgba_img
            )
        ).resize((100,100)).convert('RGB')
    )
    return rgb_img

def to_large_mask(img, cropbox):
    arr = np.array(Image.fromarray(img).resize((cropbox[3]-cropbox[2], cropbox[1]-cropbox[0])))
    new_img = np.zeros((320,480))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_img[cropbox[0]+i, cropbox[2]+j] = arr[i, j]
    return new_img

def binarize(img):
    return (np.array(
        Image.fromarray(img)
        .convert('1')
        .getdata()).reshape(img.shape[1], img.shape[0])/255).astype('uint8')


# Load the pretrained model 
model = load_model('model.h5', custom_objects={'dice_coef': dice_coef})

# Get all the filenames we need
filenames = open("filenames.txt").read().splitlines()
imagefiles = ['img_trimmed/'+ f + '_' + '{:02d}_trimmed.png'.format(i) for f in filenames for i in range(1,17)]
origfiles = [f + '_' + '{:02d}.jpg'.format(i) for f in filenames for i in range(1,17)]
medianfiles = ['mdndiff_trimmed/'+ f + '_' + '{:02d}_mdndiff.png'.format(i) for f in filenames for i in range(1,17)]

# Get the start and end indices of the batch
# from the command line
batch_start = int(sys.argv[1])
batch_end = int(sys.argv[2])

# Load data
imgdata = np.array([np.array(Image.open(fname)) for fname in imagefiles[batch_start:batch_end]])
mediandata = np.array([np.array(Image.open(fname)) for fname in medianfiles[batch_start:batch_end]])

# Transform data for neural network
imgdata_shaped = np.array([fix_size(img) for img in imgdata])/255
mdndata_shaped = np.array([fix_size(img) for img in mediandata])/255

# Set up input array
X = np.concatenate((imgdata_shaped, mdndata_shaped), axis=3)

# Normalize input in the same way as the training set
X_mean = np.array([[[[ 
    0.52757537,  
    0.51224846,  
    0.50697007,  
    0.15004876,  
    0.15179776, 
    0.14910557
]]]])

X_std = np.array([[[[ 
    0.2807387 ,  
    0.28424876,  
    0.27912231,  
    0.17270157,  
    0.17458741, 
    0.16970257
]]]])

X -= X_mean
X /= X_std

# Predict
predictions = [
    binary_fill_holes(
        to_large_mask(
            pred.reshape((100,100)), 
            get_relevant_box(img)
        )
    ) for pred, img in 
    zip((model.predict(X)>0.5).astype('uint8'), imgdata)
    ]

# Scale up to 1918x1280 for submission file
large_predictions = [
        (np.array(Image.fromarray(255*img.astype('uint8')).resize((1918, 1280)))/255).astype('uint8')
        for img in predictions
    ]

# Run-length encoding borrowed from
# https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

# Run-length-encode the predictions
rle_preds = [rle_encode(mask) for mask in large_predictions]

# Load results into dataframe
subm = pd.DataFrame(rle_preds, columns = ['rle_mask'])
subm['img'] = origfiles[batch_start:batch_end]

# Save
subm[['img', 'rle_mask']].to_csv('subm_' + str(batch_start) + '-' + str(batch_end) + '.csv', index=False)



