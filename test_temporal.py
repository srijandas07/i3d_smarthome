from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from _name_to_int import _name_to_int
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='', type=str)
parser.add_argument('--protocol', default='CS', type=str)

args = parser.parse_args()

labels = pd.read_csv("test_Labels_"+args.protocol+".csv")
y_pred = np.load('pred_arr.txt.npy')


ground_truth = open("./splits/test_"+args.protocol+".txt", "r")
lines = ground_truth.readlines()
names = [os.path.splitext(i.strip())[0] for i in lines]
y_true = [_name_to_int(os.path.splitext(i.strip())[0].split('_')[0], args.protocol) - 1 for i in lines]

accuracies = list()
gcnt = 0
pcnt = 0

while pcnt<len(y_pred):
    acc = list()
    while pcnt<len(y_pred) and labels['name'][pcnt]==names[gcnt]:
        acc.append(y_pred[pcnt].tolist())
        pcnt += 1
    accuracies.append(np.argmax(np.mean(acc, axis=0), -1))
    gcnt += 1

print(len(accuracies))
print(accuracy_score(y_true, accuracies))
print(balanced_accuracy_score(y_true, accuracies))
cnf = confusion_matrix(y_true, accuracies)
np.save("cnf.txt", cnf)
