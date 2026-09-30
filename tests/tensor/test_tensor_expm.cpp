// test_tensor_expm - Tensor expm: the matrix exponential

#include "tensor/tensor.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

void matrix_exponential(const std::string &b) {
    check(close(Tensor::zeros({3, 3}).expm(), Tensor::eye(3)), b + ": e^0 = I");
    const Tensor D({{1.0, 0.0}, {0.0, 2.0}});
    check(close(D.expm(), std::vector<double>{std::exp(1.0), 0, 0, std::exp(2.0)}, 1e-12),
          b + ": e^diag");

    // A rotation generator: e^{tJ} is a rotation by t.
    const double t = 0.7;
    const Tensor J({{0.0, 1.0}, {-1.0, 0.0}});
    check(close((J * t).expm(),
                std::vector<double>{std::cos(t), std::sin(t), -std::sin(t), std::cos(t)}, 1e-12),
          b + ": e^{tJ}");

    // e^{A} e^{-A} = I for a general A, and a large norm goes through squaring.
    const Tensor A({{0.3, -2.0, 1.0}, {1.5, 0.1, 0.0}, {0.0, -0.4, -0.2}});
    check(close((A * 6.0).expm().matmul((A * -6.0).expm()), Tensor::eye(3), 1e-8),
          b + ": e^A e^-A");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { matrix_exponential(b); });
    return test::finish("tensor expm");
}
