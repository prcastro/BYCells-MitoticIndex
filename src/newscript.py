from scipy import misc
import pandas as pd
from skimage import exposure
from skimage import transform
import numpy as np

# Load figures
table = pd.read_csv("../data/classification.csv")
table["photo"] = table.file.apply(lambda x: misc.imread("../data/" + x))

# Shuffle images
np.random.seed(0)
table = table.iloc[np.random.permutation(len(table))]

# Equalize brightness
# table["photo"] = table.photo.apply(exposure.equalize_adapthist)

# Separate into training/test
training = table.iloc[0:len(table)//2]
test = table.iloc[len(table)//2:]

# Rotate images
def rotate(df, degrees):
    result = df.copy()
    result.photo = result.photo.apply(lambda x: transform.rotate(x, degrees))
    return result

number_of_rotations = 10
orig_training = training.copy()
orig_test = test.copy()
for i in [(360/number_of_rotations) * (i+1) for i in range(number_of_rotations)]:
    training = pd.concat((training, rotate(orig_training, i)))
    test = pd.concat((test, rotate(orig_test, i)))
