# Importing Files
import os
import time
import random
import sys
import numpy as np
import nltk
import sklearn as sk
import pandas as pd
import csv
import re



# Reading the dataset
#f = open("twitter4242.txt")
#data = pd.read_csv("twitter4242.txt", sep=" ", header=None)

#print(data.columns)
f =  open("twitter4242.txt", encoding="utf8", errors='ignore')
f1 = open("output.csv", "w")
writer = csv.writer(f1)
#text_file = open("twitter4242.txt", "r")
data = f.readlines()
final_data = []

for line in data:
	line = line.strip()
	a = re.sub('[^A-Za-z0-9\t ]+', '', line)
	vals = a.split("\t")
	final_data.append(vals)
#print(final_data)
writer.writerows(final_data)

csv_data = pd.read_csv("output.csv")
print(csv_data.columns)
print(len(csv_data))
#csv_data['mean pos'] = csv_data['mean pos'].astype(int)
#csv_data['mean neg'] = csv_data['mean neg'].astype(int)
print(csv_data.iloc[1])

scores = csv_data['mean pos']/csv_data['mean neg']
csv_data['scores'] = scores
classes = []
for i in csv_data['scores']:
	if(i>1.5):
		classes.append("positive")
	elif(abs(i) == 1):
		classes.append("neutral")
	else:
		classes.append("negative")


csv_data['classes'] = classes

csv_data.to_csv("test.csv")
























