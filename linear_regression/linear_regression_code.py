import csv
import numpy as np
import matplotlib.pyplot as plt 


with open("data.csv") as f:
    f = csv.reader(f) # each row of csv file becomes list of strings
    next(f) # skips header row
    x,y = zip(*f) # unzipping using * 

print(f)