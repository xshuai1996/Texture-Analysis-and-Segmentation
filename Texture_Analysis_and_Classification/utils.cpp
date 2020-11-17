#include "Header.h"
#include <filesystem>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>


namespace fs = std::filesystem;
using cv::Ptr;
using cv::ml::SVM;
using cv::TermCriteria;
using cv::ml::ROW_SAMPLE;
using std::cout;
using std::endl;
using cv::ml::RTrees;
using std::ofstream;
using std::map;
using Eigen::SelfAdjointEigenSolver;
using std::ifstream;
using std::stringstream;
using Eigen::Success;


vector<string> iterate_folder(string path) {
	vector<string> filenames;
	for (auto& p : fs::directory_iterator(path)) {
		filenames.push_back(p.path().filename().string());
	}
	return filenames;
}


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


features_struct to_25D_features(features_struct features, bool do_normalize) {
	/* Average the feature vectors of all image pixels, leading to a 25-D feature vector for each image */
	features_struct ret = initial_features_struct(1, 1, features.num_channels, features.alias + "25D");
	float sum_value;
	// normalize by L5L5, which is the first result
	float norm_param;
	for (int k = 0; k < features.num_channels; k++) {
		sum_value = 0;
		for (int i = 0; i < features.num_rows; i++) {
			for (int j = 0; j < features.num_cols; j++) {
				sum_value += features.pointer[i][j][k];
			}
		}
		ret.pointer[0][0][k] = sum_value / (features.num_rows * features.num_cols);
		if (do_normalize == true) {
			if (k == 0) {
				norm_param = ret.pointer[0][0][k];
			}
			ret.pointer[0][0][k] /= norm_param;
		}
	}
	return ret;
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


void write_features(features_struct features, string path, int dimensions) {
	ofstream fout(path + features.alias + ".txt");
	for (int i = 0; i < dimensions; i++) {
		fout << features.pointer[0][0][i] << ",";
	}
	fout.close();
}


vector<vector<float>> standarize(vector<features_struct> data) {
	vector<float> mean_of_col;
	vector<float> deviation_of_col;
	for (int i = 0; i < data[0].num_channels; i++) {
		mean_of_col.push_back(0);
		deviation_of_col.push_back(0);
	}
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].num_channels; j++) {
			mean_of_col[j] += data[i].pointer[0][0][j];
		}
	}
	for (int j = 0; j < data[0].num_channels; j++) {
		mean_of_col[j] /= data.size();
	}
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].num_channels; j++) {
			deviation_of_col[j] += pow(data[i].pointer[0][0][j] - mean_of_col[j], 2);
		}
	}
	for (int j = 0; j < data[0].num_channels; j++) {
		deviation_of_col[j] /= data.size();
		deviation_of_col[j] = pow(deviation_of_col[j], 0.5);
	}
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].num_channels; j++) {
			data[i].pointer[0][0][j] = (data[i].pointer[0][0][j] - mean_of_col[j]) / deviation_of_col[j];
		}
	}
	vector<vector<float>> ret = { mean_of_col, deviation_of_col };
	return ret;
}


void standarize(vector<features_struct> data, vector<float> mean, vector<float> standard_devation) {
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < data[i].num_channels; j++) {
			data[i].pointer[0][0][j] = (data[i].pointer[0][0][j] - mean[j]) / standard_devation[j];
		}
	}
}


vector<features_struct> to_3D_vector_of_features(vector<features_struct> vector_of_features) {
	// build a input matrix for eigenvalue and eigenvector calculation
	MatrixXd X(vector_of_features.size(), 15);
	for (int i = 0; i < vector_of_features.size(); i++) {
		for (int j = 0; j < 15; j++) {
			X(i, j) = vector_of_features[i].pointer[0][0][j];
		}
	}

	// calculate XT*X, which is input matrix for calculate_eigen
	MatrixXd input_matrix = X.transpose() * X;
	vector<eigen_struct> eigens = calculate_eigen(input_matrix);
	vector<features_struct> ret;
	for (int i = 0; i < vector_of_features.size(); i++) {
		features_struct feature_3D = initial_features_struct(1, 1, 3,
			vector_of_features[i].alias.substr(0, vector_of_features[i].alias.length() - 3) + "3D");
		for (int j = 0; j < 3; j++) {
			feature_3D.pointer[0][0][j] = projection(vector_of_features[i], eigens[j]);
		}
		ret.push_back(feature_3D);
	}
	return ret;
}


void save_image(float_image_struct float_image, string path) {
	image_struct image = initial_image_struct(float_image.num_rows, float_image.num_cols, float_image.num_channels, float_image.alias);
	int value;
	for (int i = 0; i < image.num_rows; i++) {
		for (int j = 0; j < image.num_cols; j++) {
			for (int k = 0; k < image.num_channels; k++) {
				value = (float_image.pointer[i][j][k] < 0) ? 0 : float_image.pointer[i][j][k];
				value = (value > 255) ? 255 : value;
				image.pointer[i][j][k] = value;
			}
		}
	}
	Mat mat = image_struct_to_mat(image);
	imwrite(path + image.alias + ".jpg", mat);
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


vector<vector<features_struct>> read_features_from_directory(string path) {
	// filter the txt files end with ".txt", and catogorize them, 25D goes first, then 15D then 3D
	vector<vector<string>> filenames_vector = iterate_over_directory(path);
	map<int, int> number_of_features = { {0,25}, {1,15}, {2,3} };
	vector<vector<features_struct>> ret;
	for (int i = 0; i < 3; i++) {
		vector<features_struct> set_of_features;
		for (int j = 0; j < filenames_vector[0].size(); j++) {
			features_struct feature = initial_features_struct(1, 1, number_of_features[i],
				filenames_vector[i][j].substr(11, filenames_vector[i][j].length() - 15));
			// read file
			ifstream file(filenames_vector[i][j]);
			stringstream buffer;
			buffer << file.rdbuf();
			// split the string by comma
			for (int cnt = 0; cnt < number_of_features[i]; cnt++) {
				string substr;
				getline(buffer, substr, ',');
				feature.pointer[0][0][cnt] = stof(substr);
			}
			set_of_features.push_back(feature);
		}
		ret.push_back(set_of_features);
	}
	return ret;
}


vector<vector<features_struct>> seperate_categories_in_train_and_test(vector<features_struct> all_features) {
	vector<features_struct> test_features;
	vector<features_struct> blanket_features, brick_features, grass_features, rice_features;
	for (int i = 0; i < all_features.size(); i++) {
		if (all_features[i].alias.substr(0, 4) == "test") {
			test_features.push_back(all_features[i]);
		}
		else if (all_features[i].alias.substr(6, 7) == "blanket") {
			blanket_features.push_back(all_features[i]);
		}
		else if (all_features[i].alias.substr(6, 5) == "brick") {
			brick_features.push_back(all_features[i]);
		}
		else if (all_features[i].alias.substr(6, 5) == "grass") {
			grass_features.push_back(all_features[i]);
		}
		else if (all_features[i].alias.substr(6, 4) == "rice") {
			rice_features.push_back(all_features[i]);
		}
	}
	vector<vector<features_struct>> ret = { blanket_features, brick_features, grass_features, rice_features, test_features };
	return ret;
}


void write_scatter(vector<vector<features_struct>> data_3D, string path) {
	ofstream fout(path + "3D_gathering.txt");
	//the last one in data_3D is test data
	for (int i = 0; i < data_3D.size() - 1; i++) {
		for (int j = 0; j < data_3D[i].size(); j++) {
			fout << data_3D[i][j].alias << endl;
			for (int k = 0; k < data_3D[i][j].num_channels; k++) {
				fout << data_3D[i][j].pointer[0][0][k] << ",";
			}
			fout << endl;
		}
	}
	fout.close();
}


void K_means(vector<vector<features_struct>> seperate_features_vector) {
	vector<features_struct> train_set;
	train_set.insert(train_set.end(), seperate_features_vector[0].begin(), seperate_features_vector[0].end());
	train_set.insert(train_set.end(), seperate_features_vector[1].begin(), seperate_features_vector[1].end());
	train_set.insert(train_set.end(), seperate_features_vector[2].begin(), seperate_features_vector[2].end());
	train_set.insert(train_set.end(), seperate_features_vector[3].begin(), seperate_features_vector[3].end());
	vector<features_struct> test_set = seperate_features_vector[4];
	int changes = -1;
	int new_assignment;
	// set initial cluster centers
	vector<features_struct> centers = {
		copy_features_struct(seperate_features_vector[0][0]),
		copy_features_struct(seperate_features_vector[1][0]),
		copy_features_struct(seperate_features_vector[2][0]),
		copy_features_struct(seperate_features_vector[3][0]) };
	// set initial assignment
	int assignment[36];
	for (int i = 0; i < train_set.size(); i++) {
		assignment[i] = i % 9;
	}
	// stop K-means when assignment and centers are no longer change
	while (changes != 0) {
		changes = 0;
		// fix center, find best assignment
		for (int i = 0; i < train_set.size(); i++) {
			new_assignment = find_best_assignment(train_set[i], centers);
			if (new_assignment != assignment[i]) {
				changes += 1;
				assignment[i] = new_assignment;
			}
		}
		// fix assignment, calculate new centers
		for (int i = 0; i < 4; i++) {
			// scan assignment to find beloning data samples
			float cnt_belong = 0;
			for (int channel = 0; channel < centers[0].num_channels; channel++) {
				centers[i].pointer[0][0][channel] = 0;
			}

			for (int p = 0; p < 36; p++) {
				if (assignment[p] == i) {
					cnt_belong += 1;
					for (int channel = 0; channel < centers[0].num_channels; channel++) {
						centers[i].pointer[0][0][channel] += train_set[p].pointer[0][0][channel];
					}
				}
			}
			// get the average position
			for (int channel = 0; channel < centers[0].num_channels; channel++) {
				centers[i].pointer[0][0][channel] /= cnt_belong;
			}
		}
	}
	// predict for test set
	cout << "Assignment:" << endl;
	for (int i = 0; i < train_set.size(); i++) {
		cout << train_set[i].alias << ": " << assignment[i] << endl;
	}
	cout << "-----------------------------------------------" << endl;
	cout << "Predection:" << endl;
	vector<int> prediction;
	int best;
	for (int i = 0; i < test_set.size(); i++) {
		best = find_best_assignment(test_set[i], centers);
		cout << test_set[i].alias << ": " << best << endl;
		prediction.push_back(best);
	}
	evaluate_12(prediction);

	// free up memory
	for (int i = 0; i < 4; i++) {
		delete_features_struct(centers[i]);
	}
}


void opencv_SVM(vector<vector<features_struct>> seperate_features_vector) {
	// Set up training data
	int train_labels[36];
	for (int i = 0; i < 36; i++) {
		train_labels[i] = i / 9;
	}
	float train_data[36][3];
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			train_data[i][j] = seperate_features_vector[0][i].pointer[0][0][j];
			train_data[i + 9][j] = seperate_features_vector[1][i].pointer[0][0][j];
			train_data[i + 18][j] = seperate_features_vector[2][i].pointer[0][0][j];
			train_data[i + 27][j] = seperate_features_vector[3][i].pointer[0][0][j];
		}
	}
	Mat train_data_mat(36, 3, CV_32F, train_data);
	Mat train_labels_mat(36, 1, CV_32SC1, train_labels);

	// Set up test data
	float test_data[12][3];
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 3; j++) {
			test_data[i][j] = seperate_features_vector[4][i].pointer[0][0][j];
		}
	}

	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(train_data_mat, ROW_SAMPLE, train_labels_mat);

	cout << "Result of SVM:" << endl;
	vector<int> prediction;
	int best;
	for (int i = 0; i < 12; i++) {
		Mat test_data_mat(1, 3, CV_32F, test_data[i]);
		best = svm->predict(test_data_mat);
		cout << seperate_features_vector[4][i].alias << ": " << best << endl;
		prediction.push_back(best);
	}
	evaluate_12(prediction);
}


void evaluate_12(vector<int> prediction) {
	int ground_truth[12] = { 1, 0, 2,2,0,0,1,3,2,1,3,3 };
	float cnt = 0;
	for (int i = 0; i < 12; i++) {
		if (prediction[i] == ground_truth[i]) {
			cnt += 1;
		}
	}
	cout << "Accuracy: " << cnt / 12 * 100 << "%" << endl;
}


void opencv_RF(vector<vector<features_struct>> seperate_features_vector) {
	// Set up training data
	int train_labels[36];
	for (int i = 0; i < 36; i++) {
		train_labels[i] = i / 9;
	}
	float train_data[36][3];
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 3; j++) {
			train_data[i][j] = seperate_features_vector[0][i].pointer[0][0][j];
			train_data[i + 9][j] = seperate_features_vector[1][i].pointer[0][0][j];
			train_data[i + 18][j] = seperate_features_vector[2][i].pointer[0][0][j];
			train_data[i + 27][j] = seperate_features_vector[3][i].pointer[0][0][j];
		}
	}
	Mat train_data_mat(36, 3, CV_32F, train_data);
	Mat train_labels_mat(36, 1, CV_32SC1, train_labels);

	// Set up test data
	float test_data[12][3];
	for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 3; j++) {
			test_data[i][j] = seperate_features_vector[4][i].pointer[0][0][j];
		}
	}

	// Train the random forest
	Ptr<RTrees> rtrees = RTrees::create();
	rtrees->setMaxDepth(2);
	rtrees->setMinSampleCount(6);
	rtrees->setRegressionAccuracy(0);
	rtrees->setUseSurrogates(false);
	rtrees->setMaxCategories(4);
	rtrees->setPriors(Mat());
	rtrees->setCalculateVarImportance(true);
	rtrees->setActiveVarCount(0);
	rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 200, 1e-8));
	rtrees->train(train_data_mat, ROW_SAMPLE, train_labels_mat);

	cout << "Result of Random Forest:" << endl;
	vector<int> prediction;
	int best;
	for (int i = 0; i < 12; i++) {
		Mat test_data_mat(1, 3, CV_32F, test_data[i]);
		best = rtrees->predict(test_data_mat);
		cout << seperate_features_vector[4][i].alias << ": " << best << endl;
		prediction.push_back(best);
	}
	evaluate_12(prediction);
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


vector<vector<string>> iterate_over_directory(string path) {
	vector<string> filenames_25D, filenames_15D, filenames_3D;
	string filename;
	for (auto& p : fs::directory_iterator(path)) {
		// discard all files that not end with ".txt"
		filename = p.path().string();
		if (filename.substr(filename.length() - 3) == "txt") {
			// 25D features
			if (filename.substr(filename.length() - 7, 2) == "25") {
				filenames_25D.push_back(filename);
			}
			// 15D features
			else if (filename.substr(filename.length() - 7, 2) == "15") {
				filenames_15D.push_back(filename);
			}
			// 3D features
			else {
				filenames_3D.push_back(filename);
			}
		}
	}
	vector<vector<string>> ret = { filenames_25D, filenames_15D, filenames_3D };
	return ret;
}


float projection(features_struct feature, eigen_struct eigen) {
	float sum_eigen_square = 0, product = 0;
	for (int i = 0; i < feature.num_channels; i++) {
		sum_eigen_square += pow(eigen.eigen_vector[i], 2);
		product += feature.pointer[0][0][i] * eigen.eigen_vector[i];
	}
	float ret = product / pow(sum_eigen_square, 0.5);
	return ret;
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


features_struct copy_features_struct(features_struct obj) {
	features_struct ret = initial_features_struct(obj.num_rows, obj.num_cols, obj.num_channels, "");
	for (int i = 0; i < obj.num_rows; i++) {
		for (int j = 0; j < obj.num_cols; j++) {
			for (int k = 0; k < obj.num_channels; k++) {
				ret.pointer[i][j][k] = obj.pointer[i][j][k];
			}
		}
	}
	return ret;
}


int find_best_assignment(features_struct sample, vector<features_struct> centers) {
	int ret = -1;
	float shortest_distance = 1000000000;
	float distance;
	for (int c = 0; c < centers.size(); c++) {
		distance = 0;
		for (int i = 0; i < sample.num_channels; i++) {
			distance += pow(sample.pointer[0][0][i] - centers[c].pointer[0][0][i], 2);
		}
		if (distance < shortest_distance) {
			ret = c;
			shortest_distance = distance;
		}
	}
	return ret;
}


vector<eigen_struct> calculate_eigen(MatrixXd matrix) {
	// eigenvectors and eigenvalues
	SelfAdjointEigenSolver<MatrixXd> eigensolver(matrix);
	if (eigensolver.info() != Success) {
		abort();
	}
	// write the result back to eigen_struct
	vector<eigen_struct> ret;
	int num_eigens = eigensolver.eigenvalues().size();
	int vector_size = eigensolver.eigenvectors().size() / eigensolver.eigenvalues().size();

	for (int i = 0; i < num_eigens; i++) {
		eigen_struct eigen_value_vector;
		eigen_value_vector.eigen_value = eigensolver.eigenvalues()[i];
		eigen_value_vector.vector_size = vector_size;
		eigen_value_vector.eigen_vector = new float[vector_size];
		for (int j = 0; j < vector_size; j++) {
			eigen_value_vector.eigen_vector[j] = eigensolver.eigenvectors()(j, i);
		}
		ret.push_back(eigen_value_vector);
	}

	// return the result in descending order of eigen values
	sort(ret.begin(), ret.end(), compare_eigen_struct);
	return ret;
}


bool compare_eigen_struct(eigen_struct s1, eigen_struct s2) {
	//eigen struct with larger eigen value goes first
	return s1.eigen_value > s2.eigen_value;
}


