#include "Header.h"


int main() {
    string trainset_path = "../dataset/train";
    string testset_path = "../dataset/test";
    string result_path = "../results/";

    texture_analysis(trainset_path, testset_path, result_path);
    feature_classification(result_path, result_path);
    return 0;
}


void texture_analysis(string trainset_path, string testset_path, string result_path) {
    /* Texture Classification --- Feature Extraction */

    // iterate train and test folders and read all images
    vector<string> train_filenames = iterate_folder(trainset_path);
    vector<string> test_filenames = iterate_folder(testset_path);
    vector<float_image_struct> train_images;
    vector<float_image_struct> test_images;
    string alias;

    for (int i = 0; i < train_filenames.size(); i++) {
        // cut filename from filename.raw as alias
        alias = "train_" + train_filenames[i].substr(0, train_filenames[i].length() - 4);
        image_struct read_image = read_raw(trainset_path + "/" + train_filenames[i], 128, 128, 1, alias);
        float_image_struct float_image = image_struct_to_float_image_struct(read_image);
        subtract_global_mean(float_image);
        train_images.push_back(float_image);
        delete_image_struct(read_image);
    }

    for (int i = 0; i < test_filenames.size(); i++) {
        // cut filename from filename.raw as alias
        alias = "test_" + test_filenames[i].substr(0, test_filenames[i].length() - 4);
        image_struct read_image = read_raw(testset_path + "/" + test_filenames[i], 128, 128, 1, alias);
        float_image_struct float_image = image_struct_to_float_image_struct(read_image);
        subtract_global_mean(float_image);
        test_images.push_back(float_image);
        delete_image_struct(read_image);
    }

    // create 25 Laws' filters
    vector<filter_struct> Laws_filters = Laws_5x5_filters();

    // traverse all category, in train_images:
    // No.0-8: blanket; No.9-17: brick; No.18-26: grass; No.27-35: rice 
    float_image_struct float_image_object;
    features_struct features_25D, features_15D, features_object;
    vector<features_struct> train_features_set_25D, train_features_set_15D;
    vector<features_struct> test_features_set_25D, test_features_set_15D;

    // process train features
    for (int i = 0; i < train_images.size(); i++) {
        float_image_object = train_images[i];
        // padding and extract features
        features_object = extract_features(float_image_object, Laws_filters);
        // Average the feature vectors of all image pixels, leading to a 25-D feature vector for each image
        features_25D = to_25D_features(features_object, false);
        // Average feature vectors from pairs such as L5E5 / E5L5 to get a 15 - D feature vector
        features_15D = to_15D_features(features_25D, false);
        train_features_set_25D.push_back(features_25D);
        train_features_set_15D.push_back(features_15D);
        // record features of all training images
        write_features(features_25D, result_path, 25);
        write_features(features_15D, result_path, 15);
    }

    vector<vector<float>> mean_devation = standarize(train_features_set_15D);
    //vector<float> means = zero_center(train_features_set_15D);

    // process test features
    for (int i = 0; i < test_images.size(); i++) {
        float_image_object = test_images[i];
        // padding and extract features
        features_object = extract_features(float_image_object, Laws_filters);
        // Average the feature vectors of all image pixels, leading to a 25-D feature vector for each image
        features_25D = to_25D_features(features_object, false);
        // Average feature vectors from pairs such as L5E5 / E5L5 to get a 15 - D feature vector
        features_15D = to_15D_features(features_25D, false);
        test_features_set_25D.push_back(features_25D);
        test_features_set_15D.push_back(features_15D);
        // record features of all training images
        write_features(features_25D, result_path, 25);
        write_features(features_15D, result_path, 15);

        // free up features
        delete_features_struct(features_object);
    }

    standarize(test_features_set_15D, mean_devation[0], mean_devation[1]);
    //zero_center(train_features_set_15D, means);

    // use PCA to reduce the dimension to 3
    vector<features_struct> train_features_set_3D = to_3D_vector_of_features(train_features_set_15D);
    for (int i = 0; i < train_features_set_3D.size(); i++) {
        write_features(train_features_set_3D[i], result_path, 3);
    }

    vector<features_struct> test_features_set_3D = to_3D_vector_of_features(test_features_set_15D);
    for (int i = 0; i < test_features_set_3D.size(); i++) {
        write_features(test_features_set_3D[i], result_path, 3);
    }

    // save images and free up memory space
    for (int i = 0; i < train_images.size(); i++) {
        save_image(train_images[i], result_path);
        delete_float_image_struct(train_images[i]);
    }
    for (int i = 0; i < test_images.size(); i++) {
        save_image(test_images[i], result_path);
        delete_float_image_struct(test_images[i]);
    }
    // free up Laws' filter
    for (int i = 0; i < Laws_filters.size(); i++) {
        delete_filter_struct(Laws_filters[i]);
    }
    // free up features set
    for (int i = 0; i < train_features_set_25D.size(); i++) {
        delete_features_struct(train_features_set_25D[i]);
        delete_features_struct(train_features_set_15D[i]);
        delete_features_struct(train_features_set_3D[i]);
    }
    for (int i = 0; i < test_features_set_25D.size(); i++) {
        delete_features_struct(test_features_set_25D[i]);
        delete_features_struct(test_features_set_15D[i]);
        delete_features_struct(test_features_set_3D[i]);
    }

}


void feature_classification(string dataset_path, string result_path) {
    // read 15D, 3D features respectively from txt files
    vector<vector<features_struct>> features_vector = read_features_from_directory(dataset_path);
    // seperate read features to 0:blanket_features, 1:brick_features, 2:grass_features, 3:rice_features, 4:test_features
    vector<vector<features_struct>> seperate_15D = seperate_categories_in_train_and_test(features_vector[1]);
    vector<vector<features_struct>> seperate_3D = seperate_categories_in_train_and_test(features_vector[2]);
    // write all 3D training data in one file so that it's easy to draw scatter plot for report
    write_scatter(seperate_3D, result_path);
    K_means(seperate_15D);
    K_means(seperate_3D);
    opencv_SVM(seperate_3D);
    opencv_RF(seperate_3D);

    // delete feature structures;
    for (int i = 0; i < features_vector.size(); i++) {
        for (int j = 0; j < features_vector[0].size(); j++) {
            delete_features_struct(features_vector[i][j]);
        }
    }
}

