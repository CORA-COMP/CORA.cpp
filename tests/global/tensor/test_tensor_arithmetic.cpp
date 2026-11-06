// test_tensor_arithmetic - Tensor operations: arithmetic, matmul, elementwise, joins
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void arithmetic(const std::string &b) {
    const Tensor A({{1.0, -2.0}, {3.0, 4.0}}), B({{0.5, 1.0}, {-1.0, 2.0}});
    check(close(A + B, std::vector<double>{1.5, -1, 2, 6}), b + ": add");
    check(close(A - B, std::vector<double>{0.5, -3, 4, 2}), b + ": sub");
    check(close(A * 2.0, std::vector<double>{2, -4, 6, 8}), b + ": scale");
    check(close(0.5 * A, std::vector<double>{0.5, -1, 1.5, 2}), b + ": scale from the left");
    check(close(A.matmul(B), std::vector<double>{2.5, -3, -2.5, 11}), b + ": matmul");
    check(close(A.matrix_transpose(), std::vector<double>{1, 3, -2, 4}), b + ": transpose");
    check(A.matrix_transpose().shape() == std::vector<int64_t>({2, 2}), b + ": transpose shape");
    check(close(A.abs(), std::vector<double>{1, 2, 3, 4}), b + ": abs");
    check(close(A.maximum(0.0), std::vector<double>{1, 0, 3, 4}), b + ": positive part");
    check(close(A.minimum(0.0), std::vector<double>{0, -2, 0, 0}), b + ": negative part");

    const Tensor rect({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    check(close(rect.sum(-1, true), std::vector<double>{6, 15}), b + ": sum_last");
    check(rect.sum(-1, true).shape() == std::vector<int64_t>({2, 1}),
          b + ": sum_last keeps the axis");
    check(close(Tensor({1.0, 2.0}).diag(), std::vector<double>{1, 0, 0, 2}), b + ": diag");
    check(close(A.eye_like(), Tensor::eye(2)), b + ": eye_like");
    check(close(rect.zeros_like(), Tensor::zeros({2, 3})), b + ": zeros_like");
    // zeros and concatenation
    check(close(Tensor::concat({A, Tensor({{9.0}, {8.0}})}, -1),
                std::vector<double>{1, -2, 9, 3, 4, 8}),
          b + ": cat_last");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { arithmetic(b); });
    return test::finish("tensor arithmetic");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
