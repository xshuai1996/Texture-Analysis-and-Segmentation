#include "Header.h"


int main() {
    string dataset_path = "../dataset/";
    string result_path = "../results/";

    texture_segmentation(dataset_path, result_path);

    return 0;
}


void texture_segmentation(string dataset_path, string result_path) {
    // iterate train and test folders and read all images
    string filename = dataset_path + "comp.raw";
    image_struct image = read_raw(filename, 450, 600, 1, "COMP");
    float_image_struct float_image = image_struct_to_float_image_struct(image);
    subtract_global_mean(float_image);
    delete_image_struct(image);

    // create 25 Laws' filters
    vector<filter_struct> Laws_filters = Laws_5x5_filters();

    // calculate the features
    int window_size = 35;
    features_struct features_25D = extract_features(float_image, Laws_filters);
    features_struct features_15D = to_15D_features(features_25D, false);
    //standarize(features_15D);
    //normalize(features_15D);
    features_struct scanned_15D = scan_features_with_window(features_15D, window_size);
    features_struct features_14D = to_14D(scanned_15D);

    K_means(features_14D, result_path, window_size);

    delete_float_image_struct(float_image);
    for (int i = 0; i < Laws_filters.size(); i++) {
        delete_filter_struct(Laws_filters[i]);
    }
    delete_features_struct(features_25D);
    delete_features_struct(features_15D);
    delete_features_struct(scanned_15D);
    delete_features_struct(features_14D);
}
