import os
import numpy as np
path = os.walk("./results/")

filenames = []
for _, _, file_list in path:
    for filename in file_list:
        if filename[-7:] == '15D.txt' and filename[3:8]=='train':
             filenames.append(filename)
data = np.zeros([36, 15])

i=0
for filename in filenames:
     file = open('./results/'+filename, 'r')

     Y = [float(j) for j in file.read().split(',')[:-1]]
     for j in range(15):
          data[i,j]=Y[j]
     i+=1;

variance = []
for i in range(15):
     variance.append([i, np.var(data[:, i])])

variance.sort(key=lambda x:x[1], reverse=True)
print(variance)