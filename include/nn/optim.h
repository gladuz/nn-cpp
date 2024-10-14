#pragma once
#include<vector>
#include "tensor.h"
#include<memory>

namespace optim{
class Optimizer{
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

class SGD: public Optimizer{
public:
    SGD(std::vector<std::shared_ptr<Tensor>> parameters, double lr);
    void step() override;
    void zero_grad() override;
private:
    std::vector<std::shared_ptr<Tensor>> parameters;
    double lr;
};
}