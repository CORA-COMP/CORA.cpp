// test_neural_network_sensitivity - the Jacobian of a network at a point
//
// The sensitivity of a network is the Jacobian that central differences give, and a ReLU network
// has the slopes of its active pieces.
//
// CORA: nn/@neuralNetwork/calcSensitivity.m
// PROPERTY: nn/@neuralNetwork/calcSensitivity.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The Jacobian against central differences.
void the_sensitivity_is_the_jacobian(const std::string &b) {
    // a random network of sigmoid, tanh and identity layers and a random point
    cora::Rng rng(12);
    const auto random = [&](int64_t rows, int64_t cols, double sigma) {
        std::vector<double> data(std::size_t(rows * cols));
        rng.normal(data.data(), data.size(), sigma);
        return Tensor::from_data(data, {rows, cols});
    };
    const NeuralNetwork nn({{random(5, 3, 1.0), random(5, 1, 0.5).squeeze(-1)},
                            {random(4, 5, 1.0), random(4, 1, 0.5).squeeze(-1)},
                            {random(2, 4, 1.0), random(2, 1, 0.5).squeeze(-1)}},
                           {Activation::tanh, Activation::sigmoid, Activation::identity});
    const Tensor x = random(3, 1, 1.0).squeeze(-1);
    const Tensor J = nn.sensitivity(x);
    check(J.shape() == std::vector<int64_t>({2, 3}), b + ": the Jacobian is (out, in)");
    // central differences of the outputs, one input at a time
    const std::vector<double> j = J.data(), x0 = x.data();
    double worst = 0;
    for (int i = 0; i < 3; ++i) {
        std::vector<double> up = x0, down = x0;
        up[std::size_t(i)] += 1e-6, down[std::size_t(i)] -= 1e-6;
        const std::vector<double> yu = nn.evaluate(Tensor::from_data(up, {1, 3})).data(),
                                  yd = nn.evaluate(Tensor::from_data(down, {1, 3})).data();
        for (int o = 0; o < 2; ++o)
            worst = std::max(worst, std::abs((yu[std::size_t(o)] - yd[std::size_t(o)]) / 2e-6 -
                                             j[std::size_t(o) * 3 + std::size_t(i)]));
    }
    check(worst < 1e-7, b + ": the Jacobian agrees with finite differences");
    // a ReLU network has the slopes of the active pieces
    const NeuralNetwork relu({{Tensor({{2.0, 0.0}, {0.0, -1.0}}), Tensor({0.0, 0.0})},
                              {Tensor({{1.0, 3.0}}), Tensor({0.0})}});
    check(test::close(relu.sensitivity(Tensor({1.0, 1.0})), {2.0, 0.0}),
          b + ": the ReLU has derivative 1 where active, 0 elsewhere");
}

/// A point of another dimension is refused, and the Jacobian is the same with the checks off.
void checks_and_their_switch() {
    const NeuralNetwork net({{Tensor({{1.0, -1.0}, {0.5, 2.0}}), Tensor({0.0, 1.0})}},
                            {Activation::tanh});
    check(
        test::raises([&] { net.sensitivity(Tensor({1.0, 0.0, 0.0})); }, Error::dimension_mismatch),
        "sensitivity: a point of three dimensions");
    const Tensor x({0.3, -0.2});
    const Tensor on = net.sensitivity(x);
    ChecksScope off(false);
    check(test::close(net.sensitivity(x), on, 0), "sensitivity: the same with the checks off");
}

/// The derivative of every activation against central differences of its value.
void the_derivatives_match_differences(const std::string &b) {
    // the points keep away from the steps and from the kink of the reciprocal at 1
    const Tensor z = Tensor::from_data({0.7, 1.9, 2.4}, {3, 1});
    const double h = 1e-6;
    for (const Activation a :
         {Activation::identity, Activation::relu, Activation::sigmoid, Activation::tanh,
          Activation::sign, Activation::floor, Activation::square, Activation::cube,
          Activation::inverse, Activation::exp, Activation::reciprocal, Activation::softplus,
          Activation::root, Activation::inv_sqrt_root}) {
        // a central difference of the value at each point
        const std::vector<double> d = priv_activation_derivative(a, z).derivative.data();
        const std::vector<double> up =
            priv_activation_derivative(a, z + z.ones_like() * h).value.data();
        const std::vector<double> down =
            priv_activation_derivative(a, z - z.ones_like() * h).value.data();
        double worst = 0;
        for (std::size_t i = 0; i < 3; ++i)
            worst = std::max(worst, std::abs((up[i] - down[i]) / (2 * h) - d[i]));
        check(worst < 1e-6, b + ": the derivative of activation " + std::to_string(int(a)));
    }
    // a group sort and a product have no derivative per neuron
    check(test::raises([&] { priv_activation_derivative(Activation::group_sort, z); },
                       Error::not_supported) &&
              test::raises([&] { priv_activation_derivative(Activation::product, z); },
                           Error::not_supported),
          b + ": a group sort and a product have none");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_sensitivity_is_the_jacobian(b);
        the_derivatives_match_differences(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork sensitivity");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
