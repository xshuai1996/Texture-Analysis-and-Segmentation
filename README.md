# Texture Analysis and Segmentation

This project includes 2 parts: texture analysis and classification, and texture segmentation. Both of them use 5x5 Laws Filters to extract 25D texture features. 2D 5x5 filters are constructed by multipling 1D filters as shown below:
![image](./images/1D_Law_filters.jpg)

### Part 2: Texture Analysis and Classification

This part include from-scratch implementation that uses 36 training samples of 4 different classes (i.e. rice, grass, brick, and blanket) and classifies 12 test images. Specifically,

1. Read raw images, and extract the texture features by applying 25 Laws filters
2. Averagethe the feature vectors of all image pixels, leading to a 25D feature vector for each image, then average feature vectors from pairs such as L5E5/E5L5 to get a 15-D feature vector
3. Reduce the feature dimension from 15 to 3 using the principle component analysis (PCA)
4. Use K-means to classify 15D and 3D features, and Random Forest (RF) / Support Vector Machine (SVM) to classify 3D features.

Corresponding 15D and 3D features of each training sample are shown below:
![image](./images/texture1.jpg)
![image](./images/texture2.jpg)
![image](./images/texture3.jpg)
![image](./images/texture4.jpg)
![image](./images/texture5.jpg)
![image](./images/texture6.jpg)
![image](./images/texture7.jpg)
![image](./images/texture8.jpg)
![image](./images/texture9.jpg)
![image](./images/texture10.jpg)
![image](./images/texture11.jpg)
![image](./images/texture12.jpg)
![image](./images/texture13.jpg)
![image](./images/texture14.jpg)
![image](./images/texture15.jpg)
![image](./images/texture16.jpg)
![image](./images/texture17.jpg)
![image](./images/texture18.jpg)
![image](./images/texture19.jpg)
![image](./images/texture20.jpg)
![image](./images/texture21.jpg)
![image](./images/texture22.jpg)
Result of K-means on 3D features:
![image](./images/3D_features.jpg)
Result of classification by different methods mentioned in 4. Result of each test sample is listed for case study.
![image](./images/classify_result.jpg)

### Part 2: Texture Segmentation

In this part we segment areas with different texture inside a image:
![image](./images/original.jpg)

we segment it by steps similar to Part 1, except from:

1. Reduce the dimension of features from 15 to 14 by removing L5L5, since L5L5 is actually not good for texture segmentation and classification. L5L5's energy is used to normal all other features at each pixel
2. Use K-means to classify 14D features

With differnet size of slide windows to average the texture features inside the local cuboid, the results are different:
![image](./images/segmentation_result.jpg)


## Dependency

OpenCV and Eigen are required to run the code.

