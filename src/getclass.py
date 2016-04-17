import pickle
import os

#if os.path.isfile('classdict.p'):
#    classifs = pickle.load(open('classdict.p', 'rb'))
#else:
#    classifs = {}

classifs = {}

with open('../data/classification') as f:
    head = [f.readline() for i in range(9)]
    for line in f:
        linesp = line.split()
        classifs[linesp[0]+'.tif'] = linesp[2]

pickle.dump(classifs, open("classdict.p", "wb"))
