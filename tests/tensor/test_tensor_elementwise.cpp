// test_tensor_elementwise - Tensor::ones and the elementwise functions, on every backend

#include "testing.h"

#include <cmath>

using namespace cora::ct;
using test::check;

namespace {

void ones_fills_the_shape(const std::string &b) {
    check(test::close(Tensor::ones({2, 3}), std::vector<double>(6, 1.0)), b + ": ones (2, 3)");
    check(Tensor::ones({2, 3}).shape() == std::vector<int64_t>({2, 3}), b + ": its shape");
}

void functions_apply_to_every_element(const std::string &b) {
    const std::vector<double> x = {0.5, 1.0, 2.0, 0.25};
    const Tensor t = Tensor::fromData(x, {2, 2});
    const auto apply = [&](double (*f)(double)) {
        std::vector<double> y;
        for (const double v : x) y.push_back(f(v));
        return y;
    };
    check(test::close(t.sin(), apply(std::sin)), b + ": sin");
    check(test::close(t.cos(), apply(std::cos)), b + ": cos");
    check(test::close(t.tan(), apply(std::tan)), b + ": tan");
    check(test::close(t.exp(), apply(std::exp)), b + ": exp");
    check(test::close(t.log(), apply(std::log)), b + ": log");
    check(test::close(t.sqrt(), apply(std::sqrt)), b + ": sqrt");
    check(t.sin().shape() == t.shape(), b + ": the shape is kept");
}

// Tensor::cos(0.2) is a number, so a rotation matrix is written as in Python.
void static_functions_take_numbers_and_tensors(const std::string &b) {
    check(Tensor::sin(0.2) == std::sin(0.2) && Tensor::sqrt(4.0) == 2.0, b + ": of a number");
    const double phi = 0.2;
    const Tensor A({{Tensor::cos(phi), Tensor::sin(phi)}, {-Tensor::sin(phi), Tensor::cos(phi)}});
    check(test::close(A, std::vector<double>{std::cos(phi), std::sin(phi), -std::sin(phi),
                                             std::cos(phi)}),
          b + ": a rotation matrix");
    const Tensor t = Tensor::fromData({0.5, 2.0}, {1, 2});
    check(test::close(Tensor::exp(t), t.exp()) && test::close(Tensor::log(t), t.log()),
          b + ": of a tensor");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        ones_fills_the_shape(b);
        functions_apply_to_every_element(b);
        static_functions_take_numbers_and_tensors(b);
    });
    return test::finish("tensor elementwise");
}
