#include "nn.h"
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

    nn::Linear layer1(3, 5);
    nn::Linear layer2(5, 1);
    auto out = layer1.forward(original);
    cout<<out.rows()<<" "<<out.cols()<<endl;
    out = layer2.forward(out);
    cout<<out.rows()<<" "<<out.cols()<<endl;
    printTensor(out);
    return 0;
}

