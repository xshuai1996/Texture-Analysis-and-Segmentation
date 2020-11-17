#include "Header.h"
#include <iostream>


using std::cout;
using std::endl;
using cv::waitKey;
using cv::destroyAllWindows;
using std::to_string;


image_struct read_raw(string img_path, int num_rows, int num_cols, int num_channels, string alias) {
	image_struct image = initial_image_struct(num_rows, num_cols, num_channels, alias);
	// read img from file
	FILE* file;
	errno_t file_err = fopen_s(&file, (img_path).c_str(), "rb");
	if (file_err != 0) {
		cout << "Error: Fail to read file \"" + img_path + "\". An empty array will be returned." << endl;
	}
	else {
		int product_size = num_rows * num_cols * num_channels;
		unsigned char* flatten_img = new unsigned char[product_size];
		fread(flatten_img, sizeof(unsigned char), product_size, file);
		fclose(file);
		for (int row = 0; row < num_rows; row++) {
			for (int col = 0; col < num_cols; col++) {
				for (int channel = 0; channel < num_channels; channel++) {
					image.pointer[row][col][channel] = flatten_img[row * num_cols * num_channels + col * num_channels + channel];
				}
			}
		}
	}
	return image;
}


image_struct initial_image_struct(int num_rows, int num_cols, int num_channels, string alias) {
	image_struct new_image;
	new_image.num_rows = num_rows;
	new_image.num_cols = num_cols;
	new_image.num_channels = num_channels;
	new_image.alias = alias;
	new_image.pointer = new unsigned char** [num_rows];
	for (int row = 0; row < num_rows; row++) {
		new_image.pointer[row] = new unsigned char* [num_cols];
		for (int col = 0; col < num_cols; col++) {
			new_image.pointer[row][col] = new unsigned char[num_channels];
		}
	}
	// initialize all values to 0
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			for (int channel = 0; channel < num_channels; channel++) {
				new_image.pointer[row][col][channel] = 0;
			}
		}
	}
	return new_image;
}


float_image_struct image_struct_to_float_image_struct(image_struct image) {
	float_image_struct ret = initial_float_image_struct(image.num_rows, image.num_cols, image.num_channels, image.alias + "_float");
	for (int i = 0; i < ret.num_rows; i++) {
		for (int j = 0; j < ret.num_cols; j++) {
			for (int k = 0; k < ret.num_channels; k++) {
				ret.pointer[i][j][k] = image.pointer[i][j][k];
			}
		}
	}
	return ret;
}


float_image_struct initial_float_image_struct(int num_rows, int num_cols, int num_channels, string alias) {
	float_image_struct new_float_image;
	new_float_image.num_rows = num_rows;
	new_float_image.num_cols = num_cols;
	new_float_image.num_channels = num_channels;
	new_float_image.alias = alias;
	new_float_image.pointer = new float** [num_rows];
	for (int row = 0; row < num_rows; row++) {
		new_float_image.pointer[row] = new float* [num_cols];
		for (int col = 0; col < num_cols; col++) {
			new_float_image.pointer[row][col] = new float[num_channels];
		}
	}
	// initialize all values to 0
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			for (int channel = 0; channel < num_channels; channel++) {
				new_float_image.pointer[row][col][channel] = 0;
			}
		}
	}
	return new_float_image;
}


void subtract_global_mean(float_image_struct float_image) {
	/* as a common pre-processing step, subtract the global average of the image for each pixel location */
	float sum, average;
	for (int k = 0; k < float_image.num_channels; k++) {
		// calculate average for each channel
		sum = 0;
		for (int i = 0; i < float_image.num_rows; i++) {
			for (int j = 0; j < float_image.num_cols; j++) {
				sum += float_image.pointer[i][j][k];
			}
		}
		average = sum / (float_image.num_rows * float_image.num_cols);
		// do subtract
		for (int i = 0; i < float_image.num_rows; i++) {
			for (int j = 0; j < float_image.num_cols; j++) {
				float_image.pointer[i][j][k] -= average;
			}
		}
	}
}


void delete_image_struct(image_struct image) {
	for (int row = 0; row < image.num_rows; row++) {
		for (int col = 0; col < image.num_cols; col++) {
			delete[] image.pointer[row][col];
		}
		delete[] image.pointer[row];
	}
	delete[] image.pointer;
}


vector<filter_struct> Laws_5x5_filters() {
	int L5[] = { 1, 4, 6, 4, 1 };
	int E5[] = { -1, -2, 0, 2, 1 };
	int S5[] = { -1, 0, 2, 0, -1 };
	int W5[] = { -1, 2, 0, -2, 1 };
	int R5[] = { 1, -4, 6, -4, 1 };
	// to enable showing filter with their names, avoid using iterating
	vector<filter_struct> ret;
	filter_struct L5L5 = { tensor_product(L5, L5), "L5L5" }; ret.push_back(L5L5);
	filter_struct L5E5 = { tensor_product(L5, E5), "L5E5" }; ret.push_back(L5E5);
	filter_struct L5S5 = { tensor_product(L5, S5), "L5S5" }; ret.push_back(L5S5);
	filter_struct L5W5 = { tensor_product(L5, W5), "L5W5" }; ret.push_back(L5W5);
	filter_struct L5R5 = { tensor_product(L5, R5), "L5R5" }; ret.push_back(L5R5);

	filter_struct E5L5 = { tensor_product(E5, L5), "E5L5" }; ret.push_back(E5L5);
	filter_struct E5E5 = { tensor_product(E5, E5), "E5E5" }; ret.push_back(E5E5);
	filter_struct E5S5 = { tensor_product(E5, S5), "E5S5" }; ret.push_back(E5S5);
	filter_struct E5W5 = { tensor_product(E5, W5), "E5W5" }; ret.push_back(E5W5);
	filter_struct E5R5 = { tensor_product(E5, R5), "E5R5" }; ret.push_back(E5R5);

	filter_struct S5L5 = { tensor_product(S5, L5), "S5L5" }; ret.push_back(S5L5);
	filter_struct S5E5 = { tensor_product(S5, E5), "S5E5" }; ret.push_back(S5E5);
	filter_struct S5S5 = { tensor_product(S5, S5), "S5S5" }; ret.push_back(S5S5);
	filter_struct S5W5 = { tensor_product(S5, W5), "S5W5" }; ret.push_back(S5W5);
	filter_struct S5R5 = { tensor_product(S5, R5), "S5R5" }; ret.push_back(S5R5);

	filter_struct W5L5 = { tensor_product(W5, L5), "W5L5" }; ret.push_back(W5L5);
	filter_struct W5E5 = { tensor_product(W5, E5), "W5E5" }; ret.push_back(W5E5);
	filter_struct W5S5 = { tensor_product(W5, S5), "W5S5" }; ret.push_back(W5S5);
	filter_struct W5W5 = { tensor_product(W5, W5), "W5W5" }; ret.push_back(W5W5);
	filter_struct W5R5 = { tensor_product(W5, R5), "W5R5" }; ret.push_back(W5R5);

	filter_struct R5L5 = { tensor_product(R5, L5), "R5L5" }; ret.push_back(R5L5);
	filter_struct R5E5 = { tensor_product(R5, E5), "R5E5" }; ret.push_back(R5E5);
	filter_struct R5S5 = { tensor_product(R5, S5), "R5S5" }; ret.push_back(R5S5);
	filter_struct R5W5 = { tensor_product(R5, W5), "R5W5" }; ret.push_back(R5W5);
	filter_struct R5R5 = { tensor_product(R5, R5), "R5R5" }; ret.push_back(R5R5);
	return ret;
}


int** tensor_product(int* tensor1, int* tensor2) {
	// e.g. 1/6[1, 2, 1] product 1/2[-1, 0, 1] = 
	//	   [[-1, 0, 1],	
	// 1/12 [-2, 0, 2],
	//		[-1, 0, 1]]
	int** ret = create_int_2D_matrix(5, 5);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			ret[i][j] = tensor1[i] * tensor2[j];
		}
	}
	return ret;
}


int** create_int_2D_matrix(int num_rows, int num_cols) {
	int** ret = new int* [num_rows];
	for (int i = 0; i < num_rows; i++) {
		ret[i] = new int[num_cols];
	}
	return ret;
}


features_struct extract_features(float_image_struct float_image, vector<filter_struct> filters) {
	float_image_struct image_padding = reflection_padding_2(float_image);
	features_struct ret = initial_features_struct(float_image.num_rows, float_image.num_cols,
		filters.size() * float_image.num_channels, float_image.alias + "_features");
	float filtered_value;
	// traverse all filters
	for (int f = 0; f < filters.size(); f++) {
		for (int i = 2; i < image_padding.num_rows - 2; i++) {
			for (int j = 2; j < image_padding.num_cols - 2; j++) {
				for (int k = 0; k < image_padding.num_channels; k++) {
					filtered_value = 0;
					for (int x = 0; x < 5; x++) {
						for (int y = 0; y < 5; y++) {
							filtered_value += filters[f].pointer[x][y] * image_padding.pointer[i + x - 2][j + y - 2][k];
						}
					}
					// since positive respond and negative respond are actually the same thing, take abs value
					ret.pointer[i - 2][j - 2][f * image_padding.num_channels + k] = (filtered_value > 0) ? filtered_value : -filtered_value;
				}
			}
		}
	}
	delete_float_image_struct(image_padding);
	return ret;
}


float_image_struct reflection_padding_2(float_image_struct float_image) {
	float_image_struct ret = initial_float_image_struct(float_image.num_rows + 4,
		float_image.num_cols + 4, float_image.num_channels, float_image.alias + "_padding");
	for (int i = 0; i < float_image.num_rows; i++) {
		for (int j = 0; j < float_image.num_cols; j++) {
			for (int k = 0; k < float_image.num_channels; k++) {
				ret.pointer[i + 2][j + 2][k] = float_image.pointer[i][j][k];
			}
		}
	}
	for (int k = 0; k < ret.num_channels; k++) {
		for (int i = 2; i < ret.num_rows - 2; i++) {
			ret.pointer[i][1][k] = ret.pointer[i][2][k];
			ret.pointer[i][0][k] = ret.pointer[i][3][k];
			ret.pointer[i][ret.num_cols - 2][k] = ret.pointer[i][ret.num_cols - 3][k];
			ret.pointer[i][ret.num_cols - 1][k] = ret.pointer[i][ret.num_cols - 4][k];
		}
		for (int j = 0; j < ret.num_cols; j++) {
			ret.pointer[1][j][k] = ret.pointer[2][j][k];
			ret.pointer[0][j][k] = ret.pointer[3][j][k];
			ret.pointer[ret.num_rows - 2][j][k] = ret.pointer[ret.num_rows - 3][j][k];
			ret.pointer[ret.num_rows - 1][j][k] = ret.pointer[ret.num_rows - 4][j][k];
		}
	}
	return ret;
}


features_struct initial_features_struct(int num_rows, int num_cols, int num_channels, string alias) {
	features_struct new_features;
	new_features.num_rows = num_rows;
	new_features.num_cols = num_cols;
	new_features.num_channels = num_channels;
	new_features.alias = alias;
	new_features.pointer = new float** [num_rows];
	for (int row = 0; row < num_rows; row++) {
		new_features.pointer[row] = new float* [num_cols];
		for (int col = 0; col < num_cols; col++) {
			new_features.pointer[row][col] = new float[num_channels];
		}
	}
	return new_features;
}


void delete_float_image_struct(float_image_struct float_image) {
	for (int row = 0; row < float_image.num_rows; row++) {
		for (int col = 0; col < float_image.num_cols; col++) {
			delete[] float_image.pointer[row][col];
		}
		delete[] float_image.pointer[row];
	}
	delete[] float_image.pointer;
}


features_struct to_15D_features(features_struct features_25D, bool do_normalize) {
	/* Average feature vectors from pairs such as L5E5/E5L5 to get a 15-D feature vector */
	// set the new alias to be e.g. example_features25D -> example_features15D
	features_struct ret = initial_features_struct(features_25D.num_rows, features_25D.num_cols, 15, features_25D.alias.substr(0, features_25D.alias.length() - 3) + "15D");
	for (int i = 0; i < features_25D.num_rows; i++) {
		for (int j = 0; j < features_25D.num_cols; j++) {
			ret.pointer[i][j][0] = features_25D.pointer[0][0][0];	// L5L5
			ret.pointer[i][j][1] = 0.5 * (features_25D.pointer[i][j][1] + features_25D.pointer[i][j][5]);	// L5E5, E5L5
			ret.pointer[i][j][2] = 0.5 * (features_25D.pointer[i][j][2] + features_25D.pointer[i][j][10]);	// L5S5, S5L5
			ret.pointer[i][j][3] = 0.5 * (features_25D.pointer[i][j][3] + features_25D.pointer[i][j][15]);	// L5W5, W5L5
			ret.pointer[i][j][4] = 0.5 * (features_25D.pointer[i][j][4] + features_25D.pointer[i][j][20]);	// L5R5, R5L5
			ret.pointer[i][j][5] = features_25D.pointer[i][j][6];	// E5E5
			ret.pointer[i][j][6] = 0.5 * (features_25D.pointer[i][j][7] + features_25D.pointer[i][j][11]);	// E5S5, S5E5
			ret.pointer[i][j][7] = 0.5 * (features_25D.pointer[i][j][8] + features_25D.pointer[i][j][16]);	// E5W5, W5E5
			ret.pointer[i][j][8] = 0.5 * (features_25D.pointer[i][j][9] + features_25D.pointer[i][j][21]);	// E5R5, R5E5
			ret.pointer[i][j][9] = features_25D.pointer[i][j][12];	// S5S5
			ret.pointer[i][j][10] = 0.5 * (features_25D.pointer[i][j][13] + features_25D.pointer[i][j][17]);// S5W5, W5S5
			ret.pointer[i][j][11] = 0.5 * (features_25D.pointer[i][j][14] + features_25D.pointer[i][j][22]);// S5R5, R5S5
			ret.pointer[i][j][12] = features_25D.pointer[i][j][18]; // W5W5
			ret.pointer[i][j][13] = 0.5 * (features_25D.pointer[i][j][19] + features_25D.pointer[i][j][23]);// W5R5, R5W5
			ret.pointer[i][j][14] = features_25D.pointer[i][j][24]; // R5R5

			if (do_normalize == true) {
				// normalization, min-max, min=0, max=L5L5's value
				for (int k = 0; k < 15; k++) {
					ret.pointer[i][j][k] = ret.pointer[i][j][k] / features_25D.pointer[i][j][0];
				}
			}
		}
	}
	return ret;
}


features_struct scan_features_with_window(features_struct feature, int window_size) {
	int arm_len = (window_size - 1) / 2;
	features_struct padding = pad_feature_struct(feature, arm_len);
	features_struct ret = initial_features_struct(feature.num_rows, feature.num_cols, feature.num_channels, feature.alias + "_scanned");

	float sum_value;
	for (int k = 0; k < feature.num_channels; k++) {
		for (int i = arm_len; i < arm_len + feature.num_rows; i++) {
			for (int j = arm_len; j < arm_len + feature.num_cols; j++) {
				sum_value = 0;
				for (int x = -arm_len; x < arm_len + 1; x++) {
					for (int y = -arm_len; y < arm_len + 1; y++) {
						sum_value += padding.pointer[i + x][j + y][k];
					}
				}
				ret.pointer[i - arm_len][j - arm_len][k] = sum_value / pow(window_size, 2);
			}
		}
	}
	return ret;
}


features_struct pad_feature_struct(features_struct feature, int pad_size) {
	features_struct ret = initial_features_struct(feature.num_rows + 2 * pad_size,
		feature.num_cols + 2 * pad_size, feature.num_channels, feature.alias + "_pad");
	for (int i = 0; i < feature.num_rows; i++) {
		for (int j = 0; j < feature.num_cols; j++) {
			for (int k = 0; k < feature.num_channels; k++) {
				ret.pointer[i + pad_size][j + pad_size][k] = feature.pointer[i][j][k];
			}
		}
	}
	for (int k = 0; k < ret.num_channels; k++) {
		for (int row = pad_size; row < pad_size + feature.num_rows; row++) {
			for (int i = 0; i < pad_size; i++) {
				ret.pointer[row][pad_size - i][k] = ret.pointer[row][pad_size + i][k];
				ret.pointer[row][pad_size + feature.num_cols - 1 + i][k] = ret.pointer[row][pad_size + feature.num_cols - 1 - i][k];
			}
		}
		for (int col = 0; col < feature.num_cols + 2 * pad_size; col++) {
			for (int i = 0; i < pad_size; i++) {
				ret.pointer[pad_size - i][col][k] = ret.pointer[pad_size + i][col][k];
				ret.pointer[pad_size + feature.num_rows - 1 + i][col][k] = ret.pointer[pad_size + feature.num_rows - 1 - i][col][k];
			}
		}
	}
	return ret;
}


features_struct to_14D(features_struct feature_15D) {
	features_struct ret = initial_features_struct(feature_15D.num_rows, feature_15D.num_cols, 14, feature_15D.alias + "_14D");
	for (int i = 0; i < feature_15D.num_rows; i++) {
		for (int j = 0; j < feature_15D.num_cols; j++) {
			for (int k = 1; k < 15; k++) {
				ret.pointer[i][j][k - 1] = feature_15D.pointer[i][j][k] / feature_15D.pointer[i][j][0];
			}
		}
	}
	return ret;
}


void K_means(features_struct feature, string path, int window_size) {
	int changes = -1;
	int new_assignment;
	// set initial cluster centers
	vector<float*> centers;
	// if down sample, devide initial position by 2
	int init_pos[7][2] = { {30, 30}, {420, 30}, {30, 570}, {420, 570},{225, 300}, {225, 450}, {0, 0} };
	if (feature.num_cols != 600) {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 2; j++) {
				init_pos[i][j] /= 2;
			}
		}
	}

	for (int i = 0; i < 7; i++) {
		float* center = new float[feature.num_channels];
		for (int j = 0; j < feature.num_channels; j++) {
			center[j] = feature.pointer[init_pos[i][0]][init_pos[i][1]][j];
		}
		centers.push_back(center);
	}

	// set initial assignment
	image_struct assignment = initial_image_struct(feature.num_rows, feature.num_cols, 1, "assignment");
	for (int i = 0; i < feature.num_rows; i++) {
		for (int j = 0; j < feature.num_cols; j++) {
			assignment.pointer[i][j][0] = -1;
		}
	}

	// stop K-means when assignment and centers are no longer change
	while (changes != 0) {
		changes = 0;
		// fix center, find best assignment
		for (int i = 0; i < feature.num_rows; i++) {
			for (int j = 0; j < feature.num_cols; j++) {
				new_assignment = find_best_assignment(feature.pointer[i][j], centers, feature.num_channels);
				if (new_assignment != assignment.pointer[i][j][0]) {
					changes += 1;
					assignment.pointer[i][j][0] = new_assignment;
				}
			}
		}

		// fix assignment, calculate new centers
		for (int i = 0; i < 7; i++) {
			// scan assignment to find beloning data samples
			float cnt_belong = 0;
			for (int channel = 0; channel < feature.num_channels; channel++) {
				centers[i][channel] = 0;
			}

			for (int x = 0; x < feature.num_rows; x++) {
				for (int y = 0; y < feature.num_cols; y++) {
					if (assignment.pointer[x][y][0] == i) {
						cnt_belong += 1;
						for (int channel = 0; channel < feature.num_channels; channel++) {
							centers[i][channel] += feature.pointer[x][y][channel];
						}
					}
				}
			}
			// get the average position
			for (int channel = 0; channel < feature.num_channels; channel++) {
				centers[i][channel] /= cnt_belong;
			}
		}
		cout << "change=" << changes << endl;
	}

	// map categories to colors
	image_struct RGB_result = initial_image_struct(feature.num_rows, feature.num_cols, 3, "1cd_result" + to_string(window_size));
	int map_color[7][3] = { {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
		{255, 255, 0}, {0, 255, 255}, {255, 0, 255}, {255, 255, 255} };
	for (int i = 0; i < feature.num_rows; i++) {
		for (int j = 0; j < feature.num_cols; j++) {
			for (int k = 0; k < 3; k++) {
				RGB_result.pointer[i][j][k] = map_color[assignment.pointer[i][j][0]][k];
			}
		}
	}

	display_image(RGB_result);
	save_image(RGB_result, path);
	delete_image_struct(assignment);
	delete_image_struct(RGB_result);
}


int find_best_assignment(float* sample, vector<float*> centers, int num_channels) {
	int ret = -1;
	float shortest_distance = 10000000;
	float distance;
	for (int c = 0; c < centers.size(); c++) {
		distance = 0;
		for (int i = 0; i < num_channels; i++) {
			distance += pow(sample[i] - centers[c][i], 2);
		}
		if (distance < shortest_distance) {
			ret = c;
			shortest_distance = distance;
		}
	}
	return ret;
}


void display_image(image_struct image) {
	Mat mat = image_struct_to_mat(image);
	imshow(image.alias, mat);
	waitKey(0);
	destroyAllWindows();
}


void save_image(image_struct image, string path) {
	Mat mat = image_struct_to_mat(image);
	imwrite(path + image.alias + ".jpg", mat);
}


Mat image_struct_to_mat(image_struct image) {
	// reshape img to 1d array so that it can be Mat.data
	int product_size = image.num_rows * image.num_cols * image.num_channels;
	unsigned char* reshaped_img = new unsigned char[product_size];
	for (int row = 0; row < image.num_rows; row++) {
		for (int col = 0; col < image.num_cols; col++) {
			if (image.num_channels == 3) {	// color image
				// in opencv the rank is BGR
				reshaped_img[row * image.num_cols * image.num_channels + col * image.num_channels + 0] = image.pointer[row][col][2];
				reshaped_img[row * image.num_cols * image.num_channels + col * image.num_channels + 1] = image.pointer[row][col][1];
				reshaped_img[row * image.num_cols * image.num_channels + col * image.num_channels + 2] = image.pointer[row][col][0];
			}
			else {	// gray image
				reshaped_img[row * image.num_cols * image.num_channels + col * image.num_channels] = image.pointer[row][col][0];
			}
		}
	}
	Mat mat;
	if (image.num_channels == 3) {
		mat = Mat(image.num_rows, image.num_cols, CV_8UC3);	// CV_8U: 8-bit unsigned integer; C3: 3 channels
	}
	else {
		mat = Mat(image.num_rows, image.num_cols, CV_8UC1);
	}
	mat.data = reshaped_img;
	return mat;
}


void delete_filter_struct(filter_struct filter) {
	for (int row = 0; row < 5; row++) {
		delete[] filter.pointer[row];
	}
	delete[] filter.pointer;
}


void delete_features_struct(features_struct features) {
	for (int row = 0; row < features.num_rows; row++) {
		for (int col = 0; col < features.num_cols; col++) {
			delete[] features.pointer[row][col];
		}
		delete[] features.pointer[row];
	}
	delete[] features.pointer;
}




