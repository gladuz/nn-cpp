#include "tensor.h"
#include<stdexcept>

Tensor::Tensor(size_t rows, size_t cols): data_(rows*cols), rows_(rows), cols_(cols){}
Tensor::Tensor(const std::vector<double>& data, size_t rows, size_t cols): data_(data), rows_(rows), cols_(cols){}

double& Tensor::operator()(size_t row, size_t col){
    if(row >= rows_ || col >= cols_){
        throw std::invalid_argument("Matrix subscript is out of bounds");
    }
    return data_[row*cols_ + col];
}
double Tensor::operator()(size_t row, size_t col) const{
    if(row >= rows_ || col >= cols_){
        throw std::invalid_argument("Matrix subscript is out of bounds");
    }
    return data_[row*cols_ + col];
}

Tensor& Tensor::operator=(const Tensor& m){
    data_ = m.data_;
    rows_ = m.rows_;
    cols_ = m.cols_;
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const{
    check_sizes(other);
    Tensor result(rows_, cols_);
    /*If the rows = 1 then subscript should be adjusted*/
    int divider_broadcast = 1;
    if(other.rows_ == 1){
        divider_broadcast = cols_;
    }
    /*If rows=1, every row should be added by one column number*/
    for(size_t i=0; i<data_.size(); i++){
        result.data_[i] = data_[i] + other.data_[i%divider_broadcast];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const{
    check_sizes(other);
    Tensor result(rows_, cols_);
    /*If the rows = 1 then subscript should be adjusted*/
    int divider_broadcast = 1;
    if(other.rows_ == 1){
        divider_broadcast = cols_;
    }
    /*If rows=1, every row should be added by one column number*/
    for(size_t i=0; i<data_.size(); i++){
        result.data_[i] = data_[i] - other.data_[i%divider_broadcast];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other){
    check_sizes(other);
    for(size_t i=0; i<data_.size(); i++){
        data_[i] = data_[i] + other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    check_sizes(other);
    for(size_t i=0; i<data_.size(); i++){
        data_[i] = data_[i] - other.data_[i];
    }
    return *this;
}

Tensor Tensor::operator*(const Tensor& other) const{
    check_sizes(other);
    Tensor result(rows_, cols_);
    for(size_t i=0; i<data_.size(); i++){
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(double scalar) const{
    Tensor result(rows_, cols_);
    for(size_t i=0; i<data_.size(); i++){
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor& Tensor::operator*=(const Tensor& other){
    check_sizes(other);
    for(size_t i=0; i<data_.size(); i++){
        data_[i] = data_[i] * other.data_[i];
    }
    return *this;
}

Tensor Tensor::transpose() const{
    Tensor transposed(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            transposed(j, i) = this->operator()(i, j);
        }
    }
    return transposed;
}


Tensor& Tensor::square(){
    for(size_t i=0; i<data_.size(); i++){
        data_[i] = data_[i] * data_[i];
    }
    return *this;
}

void Tensor::zero_grad(){
    for(size_t i=0; i<grad_.size(); i++){
        grad_[i] = 0;
    }
}
Tensor Tensor::matmul(const Tensor& other) const{
    if(cols_ != other.rows_){
        throw std::invalid_argument("Number of columns and rows of the tensors should match");
    }
    Tensor result(rows_, other.cols_);
    for(auto i=0; i<rows_; i++){
        for(auto k=0; k<other.cols_; k++){
            double sum = 0;
            for(auto j=0; j<cols_; j++){                
                sum += this->operator()(i, j) * other.operator()(j, k);
            }
            result(i, k) = sum;
        }
    }
    return result;
}

void Tensor::check_sizes(const Tensor &other) const{
    if (rows_ != other.rows_ && cols_ != other.cols_){
        throw std::invalid_argument("Matrices should be in equal shape");
    }
}