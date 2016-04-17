import pickle

# import sklearn as sk
import numpy as np
from scipy.misc import imread

classifications = pickle.load(open('classdict.p', 'rb'))
files_list = list(classifications.keys())
images = np.stack([imread('../data/' + filename) for filename in files_list])
print(images.shape)
