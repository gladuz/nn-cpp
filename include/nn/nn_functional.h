#pragma once
#include "tensor/tensor.h"


namespace nn::functional{

    struct MSELossOutput {
        double loss;
        Tensor grad;
    };

    MSELossOutput mse_loss(const Tensor& predictions, const Tensor& targets);
}
