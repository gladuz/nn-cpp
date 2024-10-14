#include "nn/optim.h"

namespace optim{
    SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, double lr): parameters(parameters), lr(lr){}

    void SGD::step(){
        for(auto& param: parameters){
            for(size_t i=0; i<param->data().size(); i++){
                param->data()[i] -= lr * param->grad()[i]; 
            }
        }
    }

    void SGD::zero_grad(){
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }
}