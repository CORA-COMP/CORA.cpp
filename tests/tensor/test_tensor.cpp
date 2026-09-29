// CoraTensor on every backend: construction, the operations against their definitions, and
// the errors a caller can provoke.

#include "tensor/tensor.h"
#include "testing.h"

#include <cmath>
#include <stdexcept>

using namespace cora::ct;
using test::check;
using test::close;

namespace {

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

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
    check(close(Tensor::catLast({A, Tensor({{9.0}, {8.0}})}),
                std::vector<double>{1, -2, 9, 3, 4, 8}),
          b + ": catLast");
}

void matrix_exponential(const std::string &b) {
    check(close(Tensor::zeros({3, 3}).expm(), Tensor::eye(3)), b + ": e^0 = I");
    const Tensor D({{1.0, 0.0}, {0.0, 2.0}});
    check(close(D.expm(), std::vector<double>{std::exp(1.0), 0, 0, std::exp(2.0)}, 1e-12),
          b + ": e^diag");

    // A rotation generator: e^{tJ} is a rotation by t.
    const double t = 0.7;
    const Tensor J({{0.0, 1.0}, {-1.0, 0.0}});
    check(close((J * t).expm(), std::vector<double>{std::cos(t), std::sin(t), -std::sin(t), std::cos(t)},
                1e-12),
          b + ": e^{tJ}");

    // e^{A} e^{-A} = I for a general A, and a large norm goes through squaring.
    const Tensor A({{0.3, -2.0, 1.0}, {1.5, 0.1, 0.0}, {0.0, -0.4, -0.2}});
    check(close((A * 6.0).expm().matmul((A * -6.0).expm()), Tensor::eye(3), 1e-8),
          b + ": e^A e^-A");
}

void the_backend_is_chosen_once(const std::string &b) {
    check(close(Tensor({1.0}).data()[0], 1.0), b + ": a backend is set");
    check(throws([] { cora::ct::setBackend("jax"); }), b + ": an unknown backend");
    check(throws([] { cora::ct::setBackend("eigen:cuda"); }), b + ": eigen has no device");
    check(throws([] { cora::ct::setBackend("eigen,customBackward"); }), b + ": eigen has no options");
    check(throws([] { cora::ct::setBackend("torch,nonsense"); }), b + ": an unknown option");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        construction(b);
        arithmetic(b);
        matrix_exponential(b);
        the_backend_is_chosen_once(b);
    });
    return test::finish("tensor");
}
