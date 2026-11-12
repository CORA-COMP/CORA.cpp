// test_neural_network_regression - the regression line of an activation, and the sensitivity
//
// The line of poly_method regression (CORA's least-squares polynomial of order 1 with the error
// of computeApproxError) encloses the activation over the whole range, and its error is the
// exact maximum of the difference to the activation, which a fine grid reproduces. A range of one
// point takes the tangent.
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA: nn/layers/nonlinear/nnLeakyReLULayer.m
// CORA: nn/layers/nonlinear/nnTanhLayer.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
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

double aux_f(Activation a, double x) {
    return a == Activation::tanh      ? std::tanh(x)
           : a == Activation::sigmoid ? 1.0 / (1.0 + std::exp(-x))
                                      : std::max(x, 0.0);
}

/// One neuron a(x) over [l, u]: the slope and offset of the line, and its error radius, from the
/// zonotope the network gives (center slope c + offset, generators slope r and the error).
struct Line {
    double slope, offset, radius;
};

Line aux_line(Activation a, double l, double u) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {a});
    const double c = (l + u) / 2, r = (u - l) / 2;
    const Zonotope Y =
        nn.evaluate(Zonotope(Tensor({c}), Tensor({{r}})), {.poly_method = PolyMethod::regression});
    const std::vector<double> g = Y.generators().data();
    // the generators (1, 2): slope r, then the error (a neuron of one point has r = 0)
    const double slope = r > 0 ? g[0] / r : 0.0;
    return {slope, Y.center().data()[0] - slope * c, g.size() > 1 ? std::abs(g[1]) : 0.0};
}

/// The line encloses the activation on a fine grid, and the largest difference is the radius.
void the_line_encloses_with_the_exact_error(const std::string &b) {
    const std::vector<std::pair<double, double>> ranges{
        {-1.0, 3.0}, {-0.2, 0.1}, {0.5, 4.0}, {-6.0, -1.0}, {-3.0, 3.0}, {-20.0, 25.0}, {0.0, 2.0}};
    for (const Activation a : {Activation::tanh, Activation::sigmoid, Activation::relu}) {
        for (const auto &[l, u] : ranges) {
            const Line line = aux_line(a, l, u);
            // the stable ReLU is exact, any other range has a positive error
            double worst = 0;
            for (int k = 0; k <= 20000; ++k) {
                const double x = l + (u - l) * k / 20000.0;
                worst = std::max(worst, std::abs(aux_f(a, x) - (line.slope * x + line.offset)));
            }
            const std::string what = b + ": activation " + std::to_string(int(a)) + " on [" +
                                     std::to_string(l) + ", " + std::to_string(u) + "]";
            check(worst <= line.radius * (1 + 1e-9) + 1e-12, what + ": the line encloses");
            if (line.radius > 0)
                check(worst >= line.radius * (1 - 1e-5), what + ": the error is the maximum");
            else
                check(worst <= 1e-12, what + ": a zero error is exact");
        }
    }
}

/// The regression line of a range of one point is the tangent without error, and its slope is the
/// normal-equation slope of the 20 points.
void special_ranges(const std::string &b) {
    // a range of one point: the tangent at the point gives the value there, without error
    const Line point = aux_line(Activation::tanh, 0.7, 0.7);
    check(test::close(point.offset, std::tanh(0.7), 1e-12) && point.radius == 0.0,
          b + ": the tangent at one point");
    // a tanh over [-1, 1] is odd: the offset vanishes
    const Line odd = aux_line(Activation::tanh, -1.0, 1.0);
    check(std::abs(odd.offset) < 1e-12, b + ": an odd function has a line through the origin");
    // the least-squares slope is the normal-equation slope of the 20 points
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < 20; ++k) {
        const double x = -1.0 + 2.0 * k / 19.0, y = std::tanh(x);
        sx += x, sy += y, sxx += x * x, sxy += x * y;
    }
    check(test::close(odd.slope, (20 * sxy - sx * sy) / (20 * sxx - sx * sx), 1e-12),
          b + ": the slope of the least-squares fit");
}

/// A network of the activations with the regression method: the outputs of points lie in the set.
void the_regression_enclosure_holds_the_outputs(const std::string &b) {
    // a random network of the four activations and a random zonotope
    cora::Rng rng(11);
    const auto random = [&](int64_t rows, int64_t cols, double sigma) {
        std::vector<double> data(std::size_t(rows * cols));
        rng.normal(data.data(), data.size(), sigma);
        return Tensor::from_data(data, {rows, cols});
    };
    const NeuralNetwork nn(
        {{random(4, 3, 1.0), random(4, 1, 0.5)},
         {random(4, 4, 1.0), random(4, 1, 0.5)},
         {random(3, 4, 1.0), random(3, 1, 0.5)},
         {random(2, 3, 1.0), random(2, 1, 0.5)}},
        {Activation::tanh, Activation::sigmoid, Activation::relu, Activation::identity});
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 3);
    // the enclosures by the regression line, as a zonotope and as a polynomial zonotope
    const NnOptions regression{.poly_method = PolyMethod::regression};
    const Zonotope Y = nn.evaluate(X, regression);
    const PolyZonotope Q = nn.evaluate(PolyZonotope(X), regression);
    // the supports of both sets along random directions bound the outputs of sampled points
    int violations = 0;
    for (const std::string type : {"standard", "extreme"}) {
        const std::vector<double> y = nn.evaluate(X.rand_point(200, rng, type)).data();
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, 2);
            const double bound = test::support(Y, d), bound_poly = test::support(Q.zonotope(), d);
            for (int j = 0; j < 200; ++j) {
                violations += d[0] * y[j] + d[1] * y[200 + j] > bound + 1e-9;
                violations += d[0] * y[j] + d[1] * y[200 + j] > bound_poly + 1e-9;
            }
        }
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " outputs left the enclosure");
    // the polynomial zonotope output has the dimension of the output only
    check(Q.dim() == 2, b + ": evaluate gives the output rows");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_line_encloses_with_the_exact_error(b);
        special_ranges(b);
        the_regression_enclosure_holds_the_outputs(b);
    });
    return test::finish("neuralNetwork regression");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
