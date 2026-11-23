// test_neural_network_sign - sign as an activation (a binarized network's), on points, sets, verify
//
// CORA.cpp only: the sign activation (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// y = sign(x1) + sign(x2).
NeuralNetwork sign_network() {
    return NeuralNetwork({{Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({0.0, 0.0})},
                          {Tensor({{1.0, 1.0}}), Tensor({0.0})}},
                         {Activation::sign, Activation::identity});
}

/// The box [lo, hi]^2 and the unsafe set y <= t.
NnProperty aux_below(double lo, double hi, double t) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>{lo, lo}, {1, 2});
    p.hi = Tensor::from_data(std::vector<double>{hi, hi}, {1, 2});
    p.A = Tensor::from_data(std::vector<double>{1.0}, {1, 1, 1});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

void points_and_sets(const std::string &b) {
    const NeuralNetwork nn = sign_network();
    check(
        test::close(nn.evaluate(Tensor({{0.5, -2.0}, {0.0, 3.0}})), std::vector<double>{0.0, 1.0}),
        b + ": sign at points (sign 0 is 0)");
    // a box on one side is exact, a box across 0 gives [-1, 1] per neuron
    const Interval one =
        nn.evaluate(Zonotope(Tensor({0.5, 0.5}), Tensor({{0.25, 0.0}, {0.0, 0.25}}))).interval();
    check(test::close(one.inf(), std::vector<double>{2.0}) &&
              test::close(one.sup(), std::vector<double>{2.0}),
          b + ": a box with x > 0 maps to 2");
    const Interval across =
        nn.evaluate(Zonotope(Tensor({0.0, 0.5}), Tensor({{1.0, 0.0}, {0.0, 0.25}}))).interval();
    check(test::close(across.inf(), std::vector<double>{0.0}) &&
              test::close(across.sup(), std::vector<double>{2.0}),
          b + ": across 0 the neuron is in [-1, 1]");
}

/// verify: y <= 1.5 on [0.1, 1]^2 is unsat (y = 2), y <= -1.5 on [-1, 1]^2 is sat (the attack's
/// straight-through gradient finds x < 0).
void verify_decides(const std::string &b) {
    const NeuralNetwork nn = sign_network();
    for (const NnVerifyOptions &o : {NnVerifyOptions{}, vnncomp_options("sign")}) {
        check(nn.verify(aux_below(0.1, 1.0, 1.5), 20, o).status == "unsat",
              b + " " + o.bounding_method + ": y <= 1.5 is unsat on the positive box");
        const NnVerifyResult sat = nn.verify(aux_below(-1.0, 1.0, -1.5), 20, o);
        check(sat.status == "sat" && sat.x.size() == 2 && sat.x[0] < 0 && sat.x[1] < 0,
              b + " " + o.bounding_method + ": y <= -1.5 is sat at x < 0");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_and_sets(b);
        verify_decides(b);
    });
    return test::finish("neuralNetwork sign");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
