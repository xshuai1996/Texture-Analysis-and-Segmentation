import cv2
import numpy as np

img = cv2.imread("./results/mark.jpg", cv2.IMREAD_GRAYSCALE)

for _ in range(4):
    new_img = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            colors = set()
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if (i+x>=0 and i+x<img.shape[0] and j+y>=0 and j+y<img.shape[1]):
                        colors.add(img[i+x][j+y])
            new_img[i][j] = 255 if len(colors)>1 else img[i][j]
    img = new_img

cv2.imshow("", new_img)
cv2.waitKey()

# for _ in range(30):
#     new_img = img.copy()
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if (img[i][j]!= 255):
#                 for x in range(-1, 2):
#                     for y in range(-1, 2):
#                         if (i+x>=0 and i+x<img.shape[0] and j+y>=0 and j+y<img.shape[1] and img[i+x][j+y]==255):
#                             new_img[i + x][j + y] = img[i][j]
#     img = new_img

# cv2.imshow("", new_img)
# cv2.waitKey()