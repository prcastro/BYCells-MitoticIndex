import pickle

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

# Load images
classifications = pickle.load(open('classdict.p', 'rb'))
files_list = list(classifications.keys())
labels = list(classifications.values())

#########
# Gambiarra para elimitar as classes com menos de 2 exemplos
nlab=np.array([labels.count(l) for l in labels])
labels=np.array(labels)
labels = labels[nlab > 2]
#########
print len(labels)
images = np.stack([imread('../data/' + filename) for filename in files_list])

pca = PCA(n_components = 30)
X = np.array([np.reshape(img, (101*101)) for img in images])
X = X[nlab > 2]
X_transform = pca.fit_transform(X)
print len(X_transform)

scores=[]
for n in range(1,50):
    print n
    scores += [np.mean(cross_validation.cross_val_score(RandomForestClassifier(n_estimators = n), X_transform, labels, cv = 5))]

plt.plot(scores)
plt.show()
