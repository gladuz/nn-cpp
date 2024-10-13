#include<vector>
#include<optional>
#include "tensor.h"

namespace nn{
class Module{
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& data) = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> parameters() = 0;
protected:
    std::optional<Tensor> cache_;
};

class Linear : public Module{
public:
    Linear(size_t in_features, size_t out_features);
    Tensor forward(const Tensor& data) override;
    void backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override;
    Tensor weight_, bias_;
};
}