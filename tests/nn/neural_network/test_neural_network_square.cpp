// test_neural_network_square - x^2 as an activation, and a product of two values through squares
//
// CORA: nn/layers/nonlinear/nnSquareLayer.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"
#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// y = x1 x2 as the ONNX front end lowers it: ((x1 + x2)^2 - (x1 - x2)^2) / 4.
NeuralNetwork product_network() {
    return NeuralNetwork({{Tensor({{0.5, 0.5}, {0.5, -0.5}}), Tensor({0.0, 0.0})},
                          {Tensor({{1.0, -1.0}}), Tensor({0.0})}},
                         {Activation::square, Activation::identity});
}

/// A box (1, n) of [lo, hi] and the unsafe set y_0 <= t, or y_0 >= t with above.
NnProperty aux_half_space(int64_t n, double lo, double hi, double t, bool above) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>(std::size_t(n), lo), {1, n});
    p.hi = Tensor::from_data(std::vector<double>(std::size_t(n), hi), {1, n});
    p.A = Tensor::from_data(std::vector<double>{above ? -1.0 : 1.0}, {1, 1, 1});
    p.b = Tensor::from_data(std::vector<double>{above ? -t : t}, {1, 1});
    p.poly_case = {0};
    return p;
}

void points_are_squared(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::square});
    check(test::close(nn.evaluate(Tensor({{-1.5}, {0.5}})), std::vector<double>{2.25, 0.25}),
          b + ": x^2 at two points");
    const Tensor y = product_network().evaluate(Tensor({{3.0, -2.0}, {0.5, 0.25}}));
    check(test::close(y, std::vector<double>{-6.0, 0.125}), b + ": the product of two values");
}

/// The line of x^2 on [l, u]: every x^2 - (slope x + offset) within the radius, the radius the
/// half height (u - l)^2 / 8 of the parallelogram between the chord and the tangent.
void the_line_holds_the_square(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::square});
    int violations = 0;
    for (const auto &[l, u] : std::vector<std::pair<double, double>>{
             {-1.0, 2.0}, {0.5, 3.0}, {-4.0, -1.0}, {-1e-3, 1e-3}, {1e6, 1e6 + 1.0}}) {
        const Zonotope Y = nn.evaluate(Zonotope(Tensor({(l + u) / 2}), Tensor({{(u - l) / 2}})));
        const Interval I = Y.interval();
        for (int j = 0; j <= 100; ++j) {
            const double x = l + (u - l) * j / 100.0;
            violations += x * x < I.inf().data()[0] || x * x > I.sup().data()[0];
        }
        // the tightest parallelogram's height: the range of x^2 plus a few ulps
        const double low = l <= 0 && 0 <= u ? 0.0 : std::min(l * l, u * u);
        const double width = I.sup().data()[0] - I.inf().data()[0];
        check(width <= std::max(l * l, u * u) - low + (u - l) * (u - l) / 4 + 1e-9 * u * u + 1e-12,
              b + ": the enclosure is the parallelogram");
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " squares left the line");
}

/// A random network with squares: the zonotope encloses the sampled outputs.
void the_enclosure_contains_the_outputs(const std::string &b) {
    cora::Rng rng(7);
    const auto matrix = [&](int64_t r, int64_t c) {
        std::vector<double> data(std::size_t(r * c));
        rng.normal(data.data(), data.size(), 1.0);
        return Tensor::from_data(data, {r, c});
    };
    const NeuralNetwork nn({{matrix(4, 3), matrix(4, 1).squeeze(-1)},
                            {matrix(3, 4), matrix(3, 1).squeeze(-1)},
                            {matrix(2, 3), matrix(2, 1).squeeze(-1)}},
                           {Activation::square, Activation::relu, Activation::identity});
    // a random set and its extreme points
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const Zonotope Y = nn.evaluate(X);
    const std::vector<double> y =
        nn.evaluate(X.rand_point(300, rng, "extreme")).matrix_transpose().data();
    int violations = 0;
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        const double bound = test::support(Y, d);
        for (int j = 0; j < 300; ++j) violations += d[0] * y[j] + d[1] * y[300 + j] > bound + 1e-9;
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " outputs left the enclosure");
}

/// verify of the product on [-1, 1]^2: x1 x2 >= 1.01 is unsat (the root box's bound is [-1, 1]),
/// x1 x2 >= 0.99 is sat (a corner), x1 x2 <= -0.5 is sat; the attack's gradient goes through 2 x.
void verify_bounds_and_falsifies_the_product(const std::string &b) {
    const NeuralNetwork nn = product_network();
    // the zonotope's, the VNN-COMP preset's (alpha, CROWN) and CORA's (conzonotope) bounding
    for (const NnVerifyOptions &o :
         {NnVerifyOptions{}, vnncomp_options("square"), cora_vnncomp_options("square")}) {
        const std::string bounding = o.bounding_method;
        check(nn.verify(aux_half_space(2, -1, 1, 1.01, true), 20, o).status == "unsat",
              b + " " + bounding + ": x1 x2 >= 1.01 is unsat");
        const NnVerifyResult sat = nn.verify(aux_half_space(2, -1, 1, 0.99, true), 20, o);
        check(sat.status == "sat" && sat.x.size() == 2 && sat.x[0] * sat.x[1] >= 0.99,
              b + " " + bounding + ": x1 x2 >= 0.99 is sat at a corner");
        check(nn.verify(aux_half_space(2, -1, 1, -0.5, false), 20, o).status == "sat",
              b + " " + bounding + ": x1 x2 <= -0.5 is sat");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_are_squared(b);
        the_line_holds_the_square(b);
        the_enclosure_contains_the_outputs(b);
        verify_bounds_and_falsifies_the_product(b);
    });
    return test::finish("neuralNetwork square");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
