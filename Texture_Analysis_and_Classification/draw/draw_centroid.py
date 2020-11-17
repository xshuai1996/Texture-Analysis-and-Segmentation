import matplotlib.pyplot as plt
filename = "./results/centroids.txt"

file = open(filename, 'r')
all_info = file.read().split('\n')[:-1]
cnts = [i.split(',')[:-1] for i in all_info]

X = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

Y=[]

for i in range(4):
     new_l = [int(k) for k in cnts[i]]
     new_l = [k/sum(new_l) for k in new_l]
     Y.append(new_l)

for i in range(4):
     fig = plt.figure()
     for j in range(8):
          plt.bar(X[j], Y[i][j], color='g', width=0.5, edgecolor='black')
          # plt.text(X[j], Y[i][j]+0.05, str(Y[i][j]),
          #          horizontalalignment='center', verticalalignment='center', fontsize=7)

     plt.xlabel("features")
     plt.ylabel("count")
     plt.title("Image "+str(i+1))

     plt.savefig('./results/'+"Image "+str(i+1)+".jpg")
     plt.close()

diff = []
intersect=[]
for i in {0, 1, 3}:
     intersect.append([min(Y[i][j],Y[2][j]) for j in range(8)])
     diff.append([pow(Y[i][j]-Y[2][j], 2) for j in range(8)])

for i in range(3):
     print(sum(intersect[i]))
     print(sum(diff[i]))