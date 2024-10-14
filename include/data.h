#include <vector>
#include <random>
#include "tensor.h"

class Dataset {
public:
    Dataset(const std::string& data_path, const std::string& dataset_name);
    size_t size() const;
    std::pair<Tensor, int> get(size_t index) const;

private:
    std::vector<Tensor> data;
    std::vector<int> labels;
    void load_mnist(const std::string& data_path, bool train);
    void load_cifar10(const std::string& data_path);
};

class DataLoader{
public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = true);
    std::vector<std::pair<Tensor, int>> next_batch();
    void reset();
    bool has_next() const;
    std::pair<Tensor, Tensor> collate(std::vector<std::pair<Tensor, int>> batch);
private:
    const Dataset& dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    size_t current_index;
    std::mt19937 generator;
};