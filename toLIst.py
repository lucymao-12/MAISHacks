import numpy as np
import csv

file = open('Corona_NLP_test.csv', 'r')
file2 = open('Corona_NLP_train.csv', 'r', encoding='Latin1')

y_test_list = []
x_test_list = []

y_train_list = []
x_train_list = []

with file:
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        #print(row)
        x_test_list.append(row[4])
        y_test_list.append(row[5])

with file2:
    reader = csv.reader(file2, delimiter=',')
    for row in reader:
        x_train_list.append(row[4])
        y_train_list.append(row[5])   
    



