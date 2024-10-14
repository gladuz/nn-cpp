#include "nn/nn.h"
#include <random>
#include<cmath>

namespace nn{
Linear::Linear(size_t in_features, size_t out_features)
        : weight_(std::make_shared<Tensor>(in_features, out_features)), 
        bias_(std::make_shared<Tensor>(1, out_features)){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0, 1);
    for(auto i=0; i<in_features; i++){
        for(auto j=0; j<out_features; j++){
            (*weight_)(i, j) = d(gen) / std::sqrt(in_features);
        }
    }

    for(auto i=0; i<out_features; i++){
        (*bias_)(0, i) = 0;
    }
};
Tensor Linear::forward(const Tensor& input){
    cache_ = input;
    Tensor output = input.matmul(*weight_);
    output = output + *bias_;
    return output;
}
Tensor Linear::backward(const Tensor& grad_output){
    Tensor gradWeight = cache_->transpose().matmul(grad_output);
    weight_->grad() = gradWeight.data();
    Tensor gradBias(1, grad_output.cols());
    for(auto i=0; i<grad_output.rows(); i++){
        for(auto j=0; j<grad_output.cols(); j++){
            gradBias(0, j) += grad_output(i, j);
        }
    }
    bias_->grad() = gradBias.data();

    Tensor inputGrad = grad_output.matmul(weight_->transpose());
    return inputGrad;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters(){
    return {weight_, bias_};
}


Tensor ReLU::forward(const Tensor& input){
        Tensor output(input.rows(), input.cols());
        for (size_t i = 0; i < input.data().size(); ++i) {
            output.data()[i] = std::max(0.0, input.data()[i]);
        }
        cache_ = input;
        return output;
    }

Tensor ReLU::backward(const Tensor& grad_output){
    Tensor grad_input(grad_output.rows(), grad_output.cols());
    for (size_t i = 0; i < grad_output.data().size(); ++i) {
        grad_input.data()[i] = cache_->data()[i] > 0 ? grad_output.data()[i] : 0;
    }
    return grad_input;
}

void Sequential::add(std::shared_ptr<Module> module){
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& input){
    Tensor output = input;
    for(auto module : modules_){
    output = module->forward(output);
    }
    return output;
}

Tensor Sequential::backward(const Tensor& grad_output){
    Tensor grad = grad_output;
    for(auto it = modules_.rbegin(); it != modules_.rend(); it++){
        grad = (*it)->backward(grad);
    }
    return grad;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters(){
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& module : modules_) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    return params;
}
}


