import matplotlib.pyplot as plt
import os
path = os.walk("./results/")

filenames = []
for _, _, file_list in path:
    for filename in file_list:
        if filename[-7:] == '15D.txt':
             filenames.append(filename)

for filename in filenames:
     file = open('./results/'+filename, 'r')

     X = ["L5L5", "L5E5/E5L5", "L5S5/S5L5", "L5W5/W5L5", "L5R5/R5L5",
          "E5E5", "E5S5/S5E5", "E5W5/W5E5", "E5R5/R5E5",
          "S5S5", "S5W5/W5S5", "S5R5/R5S5",
          "W5W5", "W5R5/R5W5",
          "R5R5"]

     Y = [float(i) for i in file.read().split(',')[:-1]]
     data = [(X[i], Y[i]) for i in range(15)]

     fig = plt.figure()
     for i in data:
          plt.bar(i[0], i[1], color='g', width=0.5, edgecolor='black')
          plt.text(i[0], i[1]+0.05, str(i[1]),
                   horizontalalignment='center', verticalalignment='center', fontsize=7)

     plt.xticks(rotation=45, fontsize=6)
     plt.xlabel("filters")
     plt.ylabel("values")
     plt.title(filename[9:-22]+" "+ filename[-15:-4])

     plt.savefig('./results/'+filename[:-4]+".jpg")
     plt.close()