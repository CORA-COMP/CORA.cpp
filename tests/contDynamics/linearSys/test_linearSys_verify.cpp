// test_linearSys_verify - linearSys verify (support function algorithm) against MATLAB CORA
//
// Reference numbers: MATLAB R2025b, CORA (feature/coracpp-python), verify(sys, params, options,
// spec) with options.verifyAlg = 'reachavoid:supportFunc'; res, savedata.iterations, .timeStep,
// .nrSteps and fals.tFinal / fals.x0 are pasted below. R0 = zonotope([1;0], 0.1*eye(2)) and
// A = [-0.1 1; -1 -0.1] unless a case says otherwise; "safe" is specification(polytope(a, b),
// 'safeSet'), "unsafe" the same with 'unsafeSet'.
//
// ISS (benchmark_linear_verifyFast_ARCH23_iss_ISSF01_ISS01, 270 states, 2 unsafe specs): MATLAB
// and C++ both give res = 1, 3 iterations, timeStep 0.008, 2500 steps (checked by hand, the
// model file is not part of this repository).

#include "contDynamics/linearSys/linearSys.h"
#include "testing.h"

#include <cstdio>

using namespace cora;
using test::check;

namespace {

Tensor mat(int rows, int cols, const std::vector<double> &rowMajor) {
    return Tensor::fromData(rowMajor, {rows, cols});
}

Zonotope zono(const std::vector<double> &c, const std::vector<double> &G, int m) {
    const int n = static_cast<int>(c.size());
    return {Tensor::fromData(c, {n, 1}), Tensor::fromData(G, {n, m})};
}

Specification safe(const std::vector<double> &a, double b) {
    return Specification::safeSet(test::column(a), b);
}

Specification unsafe(const std::vector<double> &a, double b) {
    return Specification::unsafeSet(test::column(a), b);
}

const std::vector<double> Aosc{-0.1, 1, -1, -0.1};
const std::vector<double> A3{-1, -4, 0, 4, -1, 0, 0, 0, -3};
const std::vector<double> Anil{0, 1, 0, 0};

struct Expected {
    bool verified;
    int iterations;
    double timeStep;
    int nrSteps;
    double falsT;  // -1: no falsification
    std::vector<double> x0;
};

/// Checks a result against the MATLAB numbers.
void matches(const char *name, const VerifyResult &r, const Expected &e) {
    const std::string w = std::string(name) + ": ";
    check(r.verified == e.verified, w + "verified");
    check(r.iterations == e.iterations, w + "iterations " + std::to_string(r.iterations));
    check(r.nrSteps == e.nrSteps, w + "nrSteps " + std::to_string(r.nrSteps));
    check(test::close(r.timeStep, e.timeStep, 1e-12), w + "timeStep");
    check(r.fals.has_value() == (e.falsT >= 0), w + "falsification found");
    if (r.fals && e.falsT >= 0) {
        check(test::close(r.fals->tFinal, e.falsT, 1e-12), w + "falsification time");
        if (!e.x0.empty())
            check(test::close(r.fals->x0, e.x0, 1e-12), w + "falsifying initial state");
    }
}

} // namespace

int main() {
    const Zonotope R0 = zono({1, 0}, {0.1, 0, 0, 0.1}, 2);
    const Zonotope none = zono({0}, {}, 0);
    const Zonotope u05 = zono({0.05}, {0.05}, 1);
    const LinearSys osc0(mat(2, 2, Aosc), mat(2, 1, {0, 0}));
    const LinearSys osc1(mat(2, 2, Aosc), mat(2, 1, {0, 1}));
    const LinearSys nil(mat(2, 2, Anil), mat(2, 1, {0, 1}));
    const LinearSys sys3(mat(3, 3, A3), mat(3, 1, {0, 0, 1}), mat(1, 3, {1, 1, 0}));
    const Zonotope R03 = zono({1, 1, 1}, {0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1}, 3);
    const auto S = VerifyAlg::SupportFunc;
    const std::vector<double> none0;

    // Without an input set the horizon shrinks to what is not yet verified (iteration 3: 2275).
    matches("A", osc0.verify({R0, none, 5}, S, {safe({1, 0}, 1.3)}),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("J", osc0.verify({R0, none, 5}, S, {safe({0, 1}, 0.6993)}),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("J2", osc0.verify({R0, none, 5}, S, {safe({0, 1}, 0.69925)}),
            {true, 3, 0.0020000000000000005, 2275, -1, none0});
    matches("P", osc0.verify({R0, none, 5}, S, {safe({0, 1}, 0.7), safe({0, -1}, -0.7)}),
            {false, 0, 0, 0, 0, {1, -0.10000000000000001}});
    // Falsification: a hit of the unsafe set by the trajectory from the extreme point of R0.
    matches("C", osc0.verify({R0, none, 5}, S, {unsafe({1, 0}, -0.75)}),
            {false, 1, 0.050000000000000003, 100, 2.5500000000000003,
             {1.1000000000000001, -0.10000000000000001}});
    matches("M2", osc0.verify({R0, none, 5}, S, {unsafe({0, -1}, -0.65)}),
            {false, 1, 0.050000000000000003, 100, 4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("C2", osc0.verify({R0, none, 5}, S, {unsafe({1, 0}, -1.5)}),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    // With an input set (A invertible).
    matches("D", osc1.verify({R0, u05, 5}, S, {safe({1, 0}, 2)}),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("K", osc1.verify({R0, u05, 5}, S, {safe({0, 1}, 0.7958)}),
            {true, 2, 0.010000000000000002, 500, -1, none0});
    matches("K2", osc1.verify({R0, u05, 5}, S, {safe({0, 1}, 0.7943)}),
            {true, 3, 0.0020000000000000005, 2500, -1, none0});
    matches("M", osc1.verify({R0, u05, 5}, S, {unsafe({0, -1}, -0.75)}),
            {false, 1, 0.050000000000000003, 100, 4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("N", osc1.verify({R0, u05, 5}, S, {unsafe({1, 0}, -1.0)}),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("Q", osc1.verify({R0, u05, 5}, S, {safe({1, 0}, 3), unsafe({0, -1}, -0.75)}),
            {false, 1, 0.050000000000000003, 100, 4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("E", osc1.verify({R0, u05, 5}, S, {unsafe({0, 1}, 1.9)}),
            {false, 0, 0, 0, 0, {1, -0.10000000000000001}});
    // Singular A (the series of the particular solution instead of A^-1).
    matches("O", nil.verify({R0, u05, 2}, S, {safe({0, 1}, 0.3005)}),
            {true, 1, 0.02, 100, -1, none0});
    matches("G2", nil.verify({R0, u05, 2}, S, {unsafe({1, 0}, 5)}),
            {false, 0, 0, 0, 0, {0.90000000000000002, 0}});
    // An output matrix C, a three-dimensional system, several specifications.
    matches("F", sys3.verify({R03, zono({0.1}, {0.05}, 1), 4}, S, {unsafe({1}, -3), unsafe({-1}, -3)}),
            {true, 1, 0.040000000000000001, 100, -1, none0});
    matches("L", sys3.verify({R03, zono({0.1}, {0.05}, 1), 4}, S, {safe({1}, 2.202)}),
            {true, 2, 0.0080000000000000002, 500, -1, none0});

    // A sparse A (40-state chain, 7% nonzeros: sparse powers and expm). Reference: A = 5 (-2 I +
    // superdiagonal + subdiagonal) with A(1,2) = 7, R0 = zonotope(e1, [0.2 e1, 0.1 e2]), C = e10'
    // as output, B = 0 (U with a zero generator) or B = e1 with U = zonotope(0.5, 0.1), tFinal 2.
    const int n = 40;
    std::vector<double> Ach(n * n, 0.0), c0(n, 0.0), G0(n * 2, 0.0), Cch(n, 0.0), B1(n, 0.0);
    for (int i = 0; i < n; ++i) {
        Ach[i * n + i] = -10;
        if (i + 1 < n) Ach[i * n + i + 1] = 5, Ach[(i + 1) * n + i] = 5;
    }
    Ach[1] = 7;
    c0[0] = 1, G0[0] = 0.2, G0[3] = 0.1, Cch[9] = 1, B1[0] = 1;
    const Zonotope R0ch = zono(c0, G0, 2);
    const LinearSys chain0(mat(n, n, Ach), mat(n, 1, std::vector<double>(n, 0.0)), mat(1, n, Cch));
    const LinearSys chain1(mat(n, n, Ach), mat(n, 1, B1), mat(1, n, Cch));
    const Zonotope zeroGen = zono({0}, {0}, 1), u1 = zono({0.5}, {0.1}, 1);
    matches("chain safe", chain0.verify({R0ch, zeroGen, 2}, S, {safe({1}, 0.014)}),
            {true, 1, 0.02, 100, -1, none0});
    matches("chain hit", chain0.verify({R0ch, zeroGen, 2}, S, {safe({1}, 0.012)}),
            {false, 1, 0.02, 100, 1.98, none0});
    matches("chain hit 2", chain0.verify({R0ch, zeroGen, 2}, S, {safe({1}, 0.011)}),
            {false, 1, 0.02, 100, 1.84, none0});
    matches("chain input", chain1.verify({R0ch, u1, 2}, S, {safe({1}, 0.005)}),
            {false, 1, 0.02, 100, 1.16, none0});
    matches("chain input 2", chain1.verify({R0ch, u1, 2}, S, {safe({1}, 0.001)}),
            {false, 1, 0.02, 100, 0.72, none0});

    // Unsupported input is refused.
    // The support function algorithm takes unsafe sets of one halfspace only.
    check(test::throws([&] {
              osc1.verify({R0, u05, 5}, S,
                          {Specification::unsafeSet({{test::column({1, 0}), 1.0},
                                                     {test::column({0, 1}), 1.0}})});
          }),
          "an unsafe set of several halfspaces is refused by the support function algorithm");
    check(test::throws([&] { osc1.verify({R0, u05, 5}, S, {}); }), "no specification");
    check(test::throws([&] { osc1.verify({R0, u05, 5}, S, {safe({1}, 2)}); }), "wrong normal size");
    check(test::throws([&] { osc1.verify({R0, u05, 0}, S, {safe({1, 0}, 2)}); }), "tFinal 0");
    check(test::throws([&] { osc1.verify({R0, zono({0, 0}, {1, 0}, 1), 5}, S, {safe({1, 0}, 2)}); }),
          "U of the wrong dimension");
    return test::finish("linearSys verify");
}
