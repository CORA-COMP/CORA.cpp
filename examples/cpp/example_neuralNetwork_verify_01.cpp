// example_neuralNetwork_verify_01 - a set of inputs propagated through a ReLU network
//
// Affine layers map a zonotope exactly; every ReLU whose input range crosses 0 is enclosed by the
// tightest parallelogram of one slope and adds one generator. The output zonotope contains the
// outputs of all inputs: sampled outputs must stay inside it, and the property "the first output
// stays below 2" is verified from its support function.
//
// Syntax:   build/examples/cpp/example_neuralNetwork_verify_01
// Outputs:  the output set, whether the samples are inside, the verdict, and the figure as SVG
// See also: the Python example of the same name (which trains the network), example_zonotope_01

#include "global/plot/plot.h"
#include "global/rng.h"
#include "nn/neuralNetwork/neuralNetwork.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Network ---------------------------------------------------------------------------------

    // two inputs, two hidden layers of four ReLUs, two outputs
    const NeuralNetwork nn({
        {Tensor({{1.0, -0.5}, {0.8, 1.2}, {-1.1, 0.6}, {0.3, -0.9}}), Tensor({0.1, -0.2, 0.3, 0.0})},
        {Tensor({{0.9, -0.4, 0.7, 0.2}, {-0.6, 0.8, 0.1, 0.5}, {0.4, 0.4, -0.8, 0.6}, {0.2, -0.7, 0.5, 0.9}}),
         Tensor({0.0, 0.1, -0.1, 0.2})},
        {Tensor({{1.0, -1.0, 0.5, 0.3}, {0.5, 0.5, -0.4, 0.8}}), Tensor({0.0, 0.2})},
    });

    // Input Set -------------------------------------------------------------------------------

    const Zonotope X(Tensor({0.4, -0.2}), Tensor({{0.3, 0.1}, {0.0, 0.25}}));

    // Propagation -----------------------------------------------------------------------------

    const Zonotope Y = nn.evaluate(X);
    std::cout << "output set:\n" << Y;

    // Evaluation ------------------------------------------------------------------------------

    // sampled outputs must lie in the output set: check them against its support function
    cora::Rng rng(0);
    const Tensor outputs = nn.evaluate(X.randPoint(500, rng, "standard"));
    const std::vector<double> y = outputs.data();
    int outside = 0;
    for (int k = 0; k < 20; ++k) {
        std::vector<double> d(2);
        rng.normal(d.data(), d.size(), 1.0);
        const double bound = Y.supportFunc(Tensor({d[0], d[1]})).data()[0];
        for (int j = 0; j < 500; ++j) outside += d[0] * y[j] + d[1] * y[500 + j] > bound + 1e-9;
    }
    std::cout << "sampled outputs outside the output set: " << outside << " of 10000 checks\n";

    // the largest first output over the output set, from the support function along (1, 0)
    const double largest = Y.supportFunc(Tensor({1.0, 0.0})).data()[0];
    std::cout << "largest first output: " << largest
              << (largest < 2.0 ? " (verified < 2)" : " (not verified)") << "\n";

    // Visualization ---------------------------------------------------------------------------

    plot(Y, {0, 1}, {.label = "output set"});
    plot(outputs, {0, 1}, {.label = "sampled outputs"});
    figure().title = "neural network verification";
    figure().save("example_neuralNetwork_verify_01.svg");
    std::cout << "figure written to example_neuralNetwork_verify_01.svg\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
