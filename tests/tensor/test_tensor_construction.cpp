// test_tensor_construction - Tensor construction: lists, from data, zeros, eye

#include "tensor/tensor.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::column;
using test::throws;

namespace {

void construction(const std::string &b) {
    const Tensor column({1.0, 2.0, 3.0});
    check(column.shape() == std::vector<int64_t>({3, 1}), b + ": a list is a column");
    check(close(column, std::vector<double>{1, 2, 3}), b + ": the column's values");

    const Tensor matrix({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    check(matrix.shape() == std::vector<int64_t>({2, 3}), b + ": nested lists are rows");
    check(close(matrix, std::vector<double>{1, 2, 3, 4, 5, 6}), b + ": row-major values");

    check(close(Tensor::zeros({2, 3}), std::vector<double>(6, 0.0)), b + ": zeros");
    check(close(Tensor::eye(2), std::vector<double>{1, 0, 0, 1}), b + ": eye");
    check(close(Tensor::fromData({1, 2, 3, 4}, {2, 2}), Tensor({{1.0, 2.0}, {3.0, 4.0}})),
          b + ": fromData");

    check(throws([] { Tensor::fromData({1, 2, 3}, {2, 2}); }), b + ": data that misses the shape");
    check(throws([] { Tensor({{1.0, 2.0}, {3.0}}); }), b + ": ragged rows");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { construction(b); });
    return test::finish("tensor construction");
}
