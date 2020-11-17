import matplotlib.pyplot as plt
import os
path = os.walk("./results/")

filenames = []
for _, _, file_list in path:
    for filename in file_list:
        if filename[-6:] == '3D.txt':
             filenames.append(filename)

for filename in filenames:
     file = open('./results/'+filename, 'r')

     X = ["F1", "F2", "F3"]

     Y = [float(i) for i in file.read().split(',')[:-1]]
     data = [(X[i], Y[i]) for i in range(3)]

     fig = plt.figure()
     for i in data:
          plt.bar(i[0], i[1], color='g', width=0.5, edgecolor='black')
          plt.text(i[0], i[1]+0.05, str(i[1]),
                   horizontalalignment='center', verticalalignment='center', fontsize=7)

     plt.xticks(rotation=45, fontsize=6)
     plt.xlabel("filters")
     plt.ylabel("values")
     plt.title(filename[9:-21]+" "+ filename[-15:-4])

     plt.savefig('./results/'+filename[:-4]+".jpg")
     plt.close()