#ifndef HEAEDER_H
#endif HEAEDER_H


#include<string>
#include<vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


using std::string;
using std::vector;
using cv::Mat;
using Eigen::MatrixXd;



struct float_image_struct {
	float*** pointer;
	int num_rows;
	int num_cols;
	int num_channels;
	string alias;
};


struct image_struct {
	unsigned char*** pointer;
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


struct eigen_struct {
	float eigen_value;
	int vector_size;
	float* eigen_vector;
};


void texture_analysis(string trainset_path, string testset_path, string result_path);
void feature_classification(string dataset_path, string result_path);

vector<string> iterate_folder(string path);
image_struct read_raw(string img_path, int num_rows, int num_cols, int num_channels, string alias);
float_image_struct image_struct_to_float_image_struct(image_struct image);
void subtract_global_mean(float_image_struct float_image);
void delete_image_struct(image_struct image);
vector<filter_struct> Laws_5x5_filters();
features_struct extract_features(float_image_struct float_image, vector<filter_struct> filters);
features_struct to_25D_features(features_struct features, bool do_normalize);
features_struct to_15D_features(features_struct features_25D, bool do_normalize);
void write_features(features_struct features, string path, int dimensions);
vector<vector<float>> standarize(vector<features_struct> data);
void standarize(vector<features_struct> data, vector<float> mean, vector<float> standard_devation);
vector<features_struct> to_3D_vector_of_features(vector<features_struct> vector_of_features);
void save_image(float_image_struct float_image, string path);
void delete_float_image_struct(float_image_struct float_image);
void delete_filter_struct(filter_struct filter);
void delete_features_struct(features_struct features);
vector<vector<features_struct>> read_features_from_directory(string path);
vector<vector<features_struct>> seperate_categories_in_train_and_test(vector<features_struct> all_features);
void write_scatter(vector<vector<features_struct>> data_3D, string path);
void K_means(vector<vector<features_struct>> seperate_features_vector);
void opencv_SVM(vector<vector<features_struct>> seperate_features_vector);
void evaluate_12(vector<int> prediction);
void opencv_RF(vector<vector<features_struct>> seperate_features_vector);
image_struct initial_image_struct(int num_rows, int num_cols, int num_channels, string alias);
int** tensor_product(int* tensor1, int* tensor2);
int** create_int_2D_matrix(int num_rows, int num_cols);
float_image_struct reflection_padding_2(float_image_struct float_image);
float_image_struct initial_float_image_struct(int num_rows, int num_cols, int num_channels, string alias);
features_struct initial_features_struct(int num_rows, int num_cols, int num_channels, string alias);
vector<vector<string>> iterate_over_directory(string path);
float projection(features_struct feature, eigen_struct eigen);
Mat image_struct_to_mat(image_struct image);
features_struct copy_features_struct(features_struct obj);
int find_best_assignment(features_struct sample, vector<features_struct> centers);
vector<eigen_struct> calculate_eigen(MatrixXd matrix);
bool compare_eigen_struct(eigen_struct s1, eigen_struct s2);

