import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = "./results/3D_gathering.txt"

file = open(filename, 'r')
all_info = file.read().split('\n')[:-1]
info_pairs = [[all_info[i].split('_')[1][:-1], [float(j) for j in all_info[i+1].split(',')[:-1]]] for i in range(0, len(all_info), 2)]

print(info_pairs)

color_dict = {"blanket": "red", "brick": "green", "grass": "blue", "rice": "yellow"}

fig = plt.figure()
ax = Axes3D(fig)
for i in info_pairs:
    ax.scatter(i[1][0], i[1][1], i[1][2], color=color_dict[i[0]])


plt.xlabel("x")
plt.ylabel("y")
#
plt.savefig("./results/3D_features.jpg")
plt.close()