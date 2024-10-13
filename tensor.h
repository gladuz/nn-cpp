#include<vector>

class Tensor{
public:
    Tensor(const std::vector<double>& data, size_t rows, size_t cols);
    Tensor(size_t rows, size_t cols);
    
    double& operator()(size_t row, size_t col);
    double  operator() (size_t row, size_t col) const;
    Tensor& operator= (const Tensor& m);
    Tensor operator+(const Tensor &other) const;
    void check_sizes(const Tensor &other) const;
    Tensor &operator+=(const Tensor &other);
    Tensor operator*(const Tensor &other) const;
    Tensor& operator*=(const Tensor& other);
    
    // Scalar addition
    Tensor operator+(double scalar) const;
    Tensor& operator+=(double scalar);
    Tensor operator*(double scalar) const;
    Tensor& operator*=(double scalar);

    Tensor transpose() const;
    size_t rows() const{
        return rows_;
    }
    size_t cols() const{
        return cols_;
    }

    std::vector<double> data() const{
        return data_;
    }

    Tensor matmul(const Tensor& other) const;

    std::vector<double> grad;
private:
    size_t rows_, cols_;
    std::vector<double> data_;
};