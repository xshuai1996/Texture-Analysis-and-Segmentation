#ifndef HEAEDER_H
#endif HEAEDER_H


#include<string>
#include <opencv2/opencv.hpp>
#include<vector>


using std::string;
using cv::Mat;
using std::vector;


struct image_struct {
	unsigned char*** pointer;
	int num_rows;
	int num_cols;
	int num_channels;
	string alias;
};


struct float_image_struct {
	float*** pointer;
	int num_rows;
	int num_cols;
	int num_channels;
	string alias;
};


struct filter_struct {
	int** pointer;
	string name;
};


struct features_struct {
	float*** pointer;
	int num_rows;
	int num_cols;
	int num_channels;
	string alias;
};


void texture_segmentation(string dataset_path, string result_path);
image_struct read_raw(string img_path, int num_rows, int num_cols, int num_channels, string alias);
image_struct initial_image_struct(int num_rows, int num_cols, int num_channels, string alias);
float_image_struct image_struct_to_float_image_struct(image_struct image);
float_image_struct initial_float_image_struct(int num_rows, int num_cols, int num_channels, string alias);
void subtract_global_mean(float_image_struct float_image);
void delete_image_struct(image_struct image);
vector<filter_struct> Laws_5x5_filters();
int** tensor_product(int* tensor1, int* tensor2);
features_struct extract_features(float_image_struct float_image, vector<filter_struct> filters);
float_image_struct reflection_padding_2(float_image_struct float_image);
features_struct initial_features_struct(int num_rows, int num_cols, int num_channels, string alias);
void delete_float_image_struct(float_image_struct float_image);
features_struct to_15D_features(features_struct features_25D, bool do_normalize);
features_struct scan_features_with_window(features_struct feature, int window_size);
features_struct pad_feature_struct(features_struct feature, int pad_size);
features_struct to_14D(features_struct feature_15D);
void K_means(features_struct feature, string path, int window_size);
int find_best_assignment(float* sample, vector<float*> centers, int num_channels);
void display_image(image_struct image);
void save_image(image_struct image, string path);
Mat image_struct_to_mat(image_struct image);
void delete_filter_struct(filter_struct filter);
void delete_features_struct(features_struct features);
int** create_int_2D_matrix(int num_rows, int num_cols);
