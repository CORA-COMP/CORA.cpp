// test_tensor_arithmetic - Tensor operations: arithmetic, matmul, elementwise, joins

#include "global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void arithmetic(const std::string &b) {
    const Tensor A({{1.0, -2.0}, {3.0, 4.0}}), B({{0.5, 1.0}, {-1.0, 2.0}});
    check(close(A + B, std::vector<double>{1.5, -1, 2, 6}), b + ": add");
    check(close(A - B, std::vector<double>{0.5, -3, 4, 2}), b + ": sub");
    check(close(A * 2.0, std::vector<double>{2, -4, 6, 8}), b + ": scale");
    check(close(0.5 * A, std::vector<double>{0.5, -1, 1.5, 2}), b + ": scale from the left");
    check(close(A.matmul(B), std::vector<double>{2.5, -3, -2.5, 11}), b + ": matmul");
    check(close(A.transpose(), std::vector<double>{1, 3, -2, 4}), b + ": transpose");
    check(A.transpose().shape() == std::vector<int64_t>({2, 2}), b + ": transpose shape");
    check(close(A.abs(), std::vector<double>{1, 2, 3, 4}), b + ": abs");
    check(close(A.pos(), std::vector<double>{1, 0, 3, 4}), b + ": positive part");
    check(close(A.neg(), std::vector<double>{0, -2, 0, 0}), b + ": negative part");

    const Tensor rect({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    check(close(rect.sumLast(), std::vector<double>{6, 15}), b + ": sumLast");
    check(rect.sumLast().shape() == std::vector<int64_t>({2, 1}), b + ": sumLast keeps the axis");
    check(close(Tensor({1.0, 2.0}).diag(), std::vector<double>{1, 0, 0, 2}), b + ": diag");
    check(close(A.eyeLike(), Tensor::eye(2)), b + ": eyeLike");
    check(close(rect.zerosLike(), Tensor::zeros({2, 3})), b + ": zerosLike");
    check(
        close(Tensor::catLast({A, Tensor({{9.0}, {8.0}})}), std::vector<double>{1, -2, 9, 3, 4, 8}),
        b + ": catLast");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { arithmetic(b); });
    return test::finish("tensor arithmetic");
}
