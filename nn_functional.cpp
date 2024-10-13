#include "nn_functional.h"

namespace nn::functional{
    MSELossOutput mse_loss(const Tensor& predictions, const Tensor& targets){
        size_t n = predictions.rows();
        std::vector<double> squared_errors(n);
        double sum_squared_error = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double error = predictions.data()[i] - targets.data()[i];
            squared_errors[i] = error * error;
            sum_squared_error += squared_errors[i];
        }

        double loss = sum_squared_error / n;

        // Compute gradients
        Tensor grad(predictions.rows(), predictions.cols());
        double grad_scale = 2.0 / n;
        for (size_t i = 0; i < n; ++i) {
            grad.data()[i] = grad_scale * (predictions.data()[i] - targets.data()[i]);
        }

        return {loss, grad};
    }
}