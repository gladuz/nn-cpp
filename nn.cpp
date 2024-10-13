#include "nn.h"
#include <random>
#include<cmath>

namespace nn{
Linear::Linear(size_t in_features, size_t out_features): weight_(in_features, out_features), bias_(1, out_features){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, 1);
    for(auto i=0; i<in_features; i++){
        for(auto j=0; j<out_features; j++){
            weight_(i, j) = d(gen) / std::sqrt(in_features);
        }
    }

    for(auto i=0; i<out_features; i++){
        bias_(0, i) = 0;
    }
};
Tensor Linear::forward(const Tensor& input){
    cache_ = input;
    Tensor output = input.matmul(weight_);
    output = output + bias_;
    return output;
}
void Linear::backward(const Tensor& grad_output){
    Tensor gradWeight = cache_->transpose().matmul(grad_output);
    weight_.grad = gradWeight.data();
    Tensor gradBias(1, grad_output.cols());
    for(auto i=0; i<grad_output.rows(); i++){
        for(auto j=0; j<grad_output.cols(); j++){
            gradBias(0, j) += grad_output(i, j);
        }
    }
    bias_.grad = gradBias.data();
}

std::vector<Tensor*> Linear::parameters(){
    return {&weight_, &bias_};
}

}