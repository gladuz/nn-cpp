#pragma once
#include<vector>
#include<optional>
#include<memory>
#include "tensor/tensor.h"

namespace nn{

class Module{
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& data) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> parameters() { return {}; }
protected:
    std::optional<Tensor> cache_;
};

class Linear : public Module{
public:
    Linear(size_t in_features, size_t out_features);
    Tensor forward(const Tensor& data) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    std::shared_ptr<Tensor> weight_, bias_;
};

class ReLU : public Module {
public:
    ReLU() = default;
    Tensor forward(const Tensor& data) override;
    Tensor backward(const Tensor& grad_output) override;
};

class Sequential: public Module{
public:
    Sequential() = default;  
    void add(std::shared_ptr<Module> module);
    Tensor forward(const Tensor& data) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
private:
    std::vector<std::shared_ptr<Module>> modules_;
};
}
