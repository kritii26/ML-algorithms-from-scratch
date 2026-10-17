import csv
import numpy as np
import matplotlib.pyplot as plt 


with open('data.csv') as f:
    d = csv.reader(f) # each row of csv file becomes list of strings
    next(d) # skips header row
    x,y = zip(*d) # unzipping using * 

print(f)