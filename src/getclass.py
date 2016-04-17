classifs = {}
import pickle
import os


with open('../data/Classificação') as f:
    head = [f.readline() for i in range(10)]
    for line in f:
        linesp = line.split()
        classifs[linesp[0]]= linesp[2]

pickle.dump(classifs, open("classdict.p", "wb"))
