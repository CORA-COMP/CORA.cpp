// test_neural_network_floor - floor as an activation (ml4acopf's turns of an angle)
//
// On points, on sets, its line and in verify.
//
// CORA.cpp only: the floor activation (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <cmath>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// y = floor(x / (2 pi)) for one input: ml4acopf's count of turns of an angle.
NeuralNetwork turns_network() {
    return NeuralNetwork(
        {{Tensor({{1.0 / (2.0 * M_PI)}}), Tensor({0.0})}, {Tensor({{1.0}}), Tensor({0.0})}},
        {Activation::floor, Activation::identity});
}

/// The box [lo, hi] of one input and the unsafe set y <= t.
NnProperty aux_below(double lo, double hi, double t) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>{lo}, {1, 1});
    p.hi = Tensor::from_data(std::vector<double>{hi}, {1, 1});
    p.A = Tensor::from_data(std::vector<double>{1.0}, {1, 1, 1});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

void points_and_sets(const std::string &b) {
    // floor of the first neuron only (the second is the identity of x, as a floor of integers)
    const NeuralNetwork f({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::floor});
    check(test::close(f.evaluate(Tensor({{2.5}, {-0.5}, {3.0}})),
                      std::vector<double>{2.0, -1.0, 3.0}),
          b + ": floor at points");
    // a box inside one step maps to the integer, a box across steps to [floor l, floor u]
    const Interval one = f.evaluate(Zonotope(Tensor({1.5}), Tensor({{0.25}}))).interval();
    check(test::close(one.inf(), std::vector<double>{1.0}) &&
              test::close(one.sup(), std::vector<double>{1.0}),
          b + ": a box inside [1, 2) maps to 1");
    const Interval across = f.evaluate(Zonotope(Tensor({1.0}), Tensor({{1.5}}))).interval();
    check(test::close(across.inf(), std::vector<double>{-1.0}) &&
              test::close(across.sup(), std::vector<double>{2.0}),
          b + ": [-0.5, 2.5] maps to [-1, 2]");
}

/// The line encloses floor at sampled points of random boxes.
void line_holds(const std::string &b) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-5.0, 5.0), V(0.0, 1.0);
    int violations = 0;
    for (int k = 0; k < 50; ++k) {
        const double l = U(rng), u = l + 3.0 * V(rng);
        const PrivLine line = priv_floor_line(Tensor({l}), Tensor({u}));
        const double o = line.offset.reshape({1}).item(), r = line.radius.reshape({1}).item();
        // slope 0: every floor value of the box inside [o - r, o + r]
        for (int s = 0; s <= 20; ++s) {
            const double y = std::floor(l + (u - l) * s / 20.0);
            violations += y < o - r || y > o + r;
        }
        violations += line.slope.reshape({1}).item() != 0.0;
    }
    check(violations == 0, b + ": floor's line holds 1050 points of 50 boxes");
}

/// verify: on [5, 8] the turns are 0 or 1, so y <= -0.5 is unsat (across a step) and y <= 0.5 is
/// sat below 2 pi.
void verify_decides(const std::string &b) {
    const NeuralNetwork nn = turns_network();
    for (const NnVerifyOptions &o : {NnVerifyOptions{}, vnncomp_options("floor")}) {
        check(nn.verify(aux_below(5.0, 8.0, -0.5), 20, o).status == "unsat",
              b + " " + o.bounding_method + ": y <= -0.5 is unsat on [5, 8]");
        const NnVerifyResult sat = nn.verify(aux_below(5.0, 8.0, 0.5), 20, o);
        check(sat.status == "sat" && sat.x.size() == 1 && sat.x[0] < 2.0 * M_PI,
              b + " " + o.bounding_method + ": y <= 0.5 is sat below 2 pi");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_and_sets(b);
        line_holds(b);
        verify_decides(b);
    });
    return test::finish("neuralNetwork floor");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
