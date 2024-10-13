#include "nn.h"
#include "nn_functional.h"
#include "optim.h"
#include <iostream>
using namespace std;

void printTensor(Tensor &data)
{
    for (size_t i = 0; i < data.rows(); ++i)
    {
        for (size_t j = 0; j < data.cols(); ++j)
        {
            cout << data(i, j) << " ";
        }
        cout << endl;
    }
}


int main() {
    Tensor original({1, 2, 3, 4, 5, 6}, 2, 3);  // 2x3 matrix
    
    cout << "Original matrix:" << endl;
    printTensor(original);
    
    Tensor transposed = original.transpose();
    
    cout << "Transposed matrix:" << endl;
    printTensor(transposed);

    Tensor broadTens({1,2,3}, 1, 3);
    Tensor broadRes = original + broadTens;
    printTensor(broadRes);


    nn::Sequential seq;
    seq.add(std::make_shared<nn::Linear>(3, 5));
    seq.add(std::make_shared<nn::ReLU>());
    seq.add(std::make_shared<nn::Linear>(5, 7));
    seq.add(std::make_shared<nn::ReLU>());
    seq.add(std::make_shared<nn::Linear>(7, 1));

    auto optimizer = optim::SGD(seq.parameters(), 0.01);

    Tensor corr({2, -2}, 2, 1);

    for(int i=0; i<100; i++){
        auto out = seq.forward(original);
        auto [loss, loss_grad] = nn::functional::mse_loss(out, corr);
        cout<<"Loss at epoch: "<<i<<" is: "<<loss<<endl;
        seq.backward(loss_grad);
        optimizer.step();
    }
    
    return 0;
}

