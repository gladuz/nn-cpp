#include "data.h"
#include <fstream>
#include <iostream>

Dataset::Dataset(const std::string& data_path, const std::string& dataset_name){
    if(dataset_name == "mnist"){
        load_mnist(data_path, true);
    }
}

size_t Dataset::size() const {
    return data.size();
}

std::pair<Tensor, int> Dataset::get(size_t index) const {
    return {data[index], labels[index]};
}

int32_t read_int32(std::ifstream& file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return __builtin_bswap32(value);  // Convert from big-endian to little-endian
}

void Dataset::load_mnist(const std::string& data_folder, bool train) {
    std::string labels_path = train ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
    std::string images_path = train ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
    labels_path = data_folder + "/" + labels_path;
    images_path = data_folder + "/" + images_path;

    std::ifstream data_file(images_path, std::ios::binary);
    if (!data_file) {
        std::cerr << "Cannot open file: " << images_path << std::endl;
        return;
    }

    int32_t magic_number = read_int32(data_file);
    int32_t num_images = read_int32(data_file);
    int32_t num_rows = read_int32(data_file);
    int32_t num_cols = read_int32(data_file);

    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file!" << std::endl;
        return;
    }

    // Read labels
    std::ifstream label_file(labels_path, std::ios::binary);
    if (!label_file) {
        std::cerr << "Cannot open file: " << labels_path << std::endl;
        return;
    }

    magic_number = read_int32(label_file);
    int32_t num_labels = read_int32(label_file);

    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file!" << std::endl;
        return;
    }

    if (num_images != num_labels) {
        std::cerr << "Number of labels and images don't match!" << std::endl;
        return;
    }

    // Read image data
    std::vector<uint8_t> pixels(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        data_file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        
        // Convert to float and normalize to [0, 1]
        std::vector<double> float_pixels(pixels.size());
        for (size_t j = 0; j < pixels.size(); ++j) {
            float_pixels[j] = static_cast<double>(pixels[j]) / 255.0f;
        }
        
        data.push_back(Tensor(float_pixels, num_rows, num_cols));

        // Read corresponding label
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }

    std::cout << "Loaded " << num_images << " images" << std::endl;
}