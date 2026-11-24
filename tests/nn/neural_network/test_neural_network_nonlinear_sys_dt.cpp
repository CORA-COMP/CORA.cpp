// test_neural_network_nonlinear_sys_dt - a network as a discrete-time system
//
// The state stays, and the output is the network's with an offset per layer neuron added to the
// biases: f(x, 0) is the network, C is its Jacobian, and D is the Jacobian of the output by the
// offsets, which the tests compare with central differences. CORALean has no such system, so
// these are property tests.
//
// CORA: nn/@neuralNetwork/nonlinearSysDT.m
// PROPERTY: nn/@neuralNetwork/nonlinearSysDT.m
// CORA authors: Laura Luetzow
// Authors:       Claude Sonnet 5.5
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

NeuralNetwork aux_network() {
    return NeuralNetwork(
        {{Tensor({{1.0, -2.0}, {0.5, 1.5}, {-1.0, 1.0}}), Tensor({0.1, -0.2, 0.3})},
         {Tensor({{1.0, 1.0, -1.0}, {0.5, -1.0, 2.0}}), Tensor({0.0, 0.2})}},
        {Activation::tanh, Activation::sigmoid});
}

void system(const std::string &b) {
    const NeuralNetwork nn = aux_network();
    const NetworkSystemDT sys = nn.nonlinear_sys_dt();
    check(sys.num_states() == 2 && sys.num_inputs() == 5 && sys.num_outputs() == 2,
          b + ": the dimensions");
    // a point from a fixed seed
    Rng rng(4);
    std::vector<double> point(2);
    rng.uniform(point.data(), 2, -1.0, 1.0);
    const Tensor x = Tensor::from_data(point, {2}), zero = Tensor::zeros({5});
    // the output at u = 0 is the network, the state stays
    check(test::close(sys.output(x, zero), nn.evaluate(x.expand_dims(0)).squeeze(0), 1e-12),
          b + ": f(x, 0) is the network");
    check(test::close(sys.next_state(x, zero), x, 0.0), b + ": the state stays");
    check(test::close(sys.output_jacobian_x(x), nn.sensitivity(x), 0.0),
          b + ": C is the sensitivity");
    // D against central differences in each offset
    const Tensor u({0.1, -0.2, 0.05, 0.3, -0.1});
    const Tensor D = sys.output_jacobian_u(x, u);
    check(D.shape() == std::vector<int64_t>{2, 5}, b + ": D is (outputs, inputs)");
    const std::vector<double> d = D.data(), uv = u.data();
    bool match = true;
    for (std::size_t j = 0; j < 5; ++j) {
        std::vector<double> up = uv, down = uv;
        up[j] += 1e-6;
        down[j] -= 1e-6;
        const std::vector<double> yp = sys.output(x, Tensor::from_data(up, {5})).data(),
                                  ym = sys.output(x, Tensor::from_data(down, {5})).data();
        for (std::size_t i = 0; i < 2; ++i)
            match = match && std::abs((yp[i] - ym[i]) / 2e-6 - d[i * 5 + j]) < 1e-7;
    }
    check(match, b + ": D matches the central differences");
    // a ReLU layer, and the checks
    const NeuralNetwork relu({{Tensor({{1.0, 1.0}}), Tensor({0.0})}}, {Activation::relu});
    check(test::close(relu.nonlinear_sys_dt().output_jacobian_u(x, Tensor({0.0})), {0.0}, 0.0),
          b + ": a ReLU of a negative sum has no sensitivity");
    check(test::raises([&] { sys.output(Tensor({0.0}), zero); }, Error::dimension_mismatch),
          b + ": a wrong state");
    check(test::raises([&] { sys.next_state(Tensor({0.0}), zero); }, Error::dimension_mismatch),
          b + ": a wrong state for the next state");
    check(test::raises([&] { sys.output_jacobian_u(x, Tensor({0.0})); }, Error::dimension_mismatch),
          b + ": a wrong input for D");
    check(test::raises([&] { NeuralNetwork(NeuralNetwork::exp_layer(2)).nonlinear_sys_dt(); },
                       Error::not_supported),
          b + ": an exp layer");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(sys.output(x, zero).shape()[0] == 2, b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { system(b); });
    return test::finish("neuralNetwork nonlinearSysDT");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
