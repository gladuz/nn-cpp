#include "data.h"
#include <algorithm>

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_index(0){
    indices.resize(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        indices[i] = i;
    }
    reset();
}

std::vector<std::pair<Tensor, int>> DataLoader::next_batch() {
    std::vector<std::pair<Tensor, int>> batch;
    for (size_t i = 0; i < batch_size && current_index < dataset.size(); ++i, ++current_index) {
        batch.push_back(dataset.get(indices[current_index]));
    }
    return batch;
}

std::pair<Tensor, Tensor> collate(std::vector<std::pair<Tensor, int>> batch){
    
}

void DataLoader::reset() {
    current_index = 0;
    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), generator);
    }
}

bool DataLoader::has_next() const {
    return current_index < dataset.size();
}