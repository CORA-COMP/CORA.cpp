// test_linear_sys_verify - linear_sys verify (support function algorithm) against MATLAB CORA
//
// Reference numbers: MATLAB R2025b, CORA (feature/coracpp-python), verify(sys, params, options,
// spec) with options.verifyAlg = 'reachavoid:support_function'; res, savedata.iterations,
// .time_step, .num_steps and falsification.t_final / falsification.x0 are pasted below. R0 =
// zonotope([1;0], 0.1*eye(2)) and A = [-0.1 1; -1 -0.1] unless a case says otherwise; "safe" is
// specification(polytope(a, b), 'safe_set'), "unsafe" the same with 'unsafe_set'.
//
// ISS (benchmark_linear_verifyFast_ARCH23_iss_ISSF01_ISS01, 270 states, 2 unsafe specs): MATLAB
// and C++ both give res = 1, 3 iterations, time_step 0.008, 2500 steps (checked by hand, the
// model file is not part of this repository).
//
// CORA: contDynamics/@linearSys/verify.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "spec_testing.h"
#include "testing.h"

#include <cstdio>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Tensor mat(int rows, int cols, const std::vector<double> &row_major) {
    return Tensor::from_data(row_major, {rows, cols});
}

Zonotope zono(const std::vector<double> &c, const std::vector<double> &G, int m) {
    const int n = static_cast<int>(c.size());
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, m})};
}

Specification safe(const std::vector<double> &a, double b) {
    return Specification::safe_set(test::halfspace(test::column(a), b));
}

Specification unsafe(const std::vector<double> &a, double b) {
    return Specification::unsafe_set(test::halfspace(test::column(a), b));
}

const std::vector<double> Aosc{-0.1, 1, -1, -0.1};
const std::vector<double> A3{-1, -4, 0, 4, -1, 0, 0, 0, -3};
const std::vector<double> Anil{0, 1, 0, 0};

struct Expected {
    bool verified;
    int iterations;
    // the expected result
    double time_step;
    int num_steps;
    double fals_t; // -1: no falsification
    std::vector<double> x0;
};

/// Checks a result against the MATLAB numbers.
void matches(const char *name, const VerifyResult &r, const Expected &e) {
    const std::string w = std::string(name) + ": ";
    check(r.verified == e.verified, w + "verified");
    check(r.iterations == e.iterations, w + "iterations " + std::to_string(r.iterations));
    check(r.num_steps == e.num_steps, w + "num_steps " + std::to_string(r.num_steps));
    check(test::close(r.time_step, e.time_step, 1e-12), w + "time_step");
    check(r.falsification.has_value() == (e.fals_t >= 0), w + "falsification found");
    if (r.falsification && e.fals_t >= 0) {
        check(test::close(r.falsification->time, e.fals_t, 1e-12), w + "falsification time");
        if (!e.x0.empty())
            check(test::close(r.falsification->x0, e.x0, 1e-12), w + "falsifying initial state");
    }
}

/// t_final 0 and no specification are refused; with the checks off the verdict is the same.
void checks_and_their_switch() {
    // a system with an input, and an initial set around (1, 1)
    const LinearSys sys = LinearSys(Tensor({{-1.0, 0.5}, {-0.5, -1.0}}), Tensor({{1.0}, {0.0}}));
    const ReachProblem problem{.initial_set =
                                   Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}})),
                               .input_set = Zonotope(Tensor({0.0}), Tensor({{0.1}})),
                               .t_final = 1};
    const std::vector<Specification> specs{
        Specification::safe_set(test::halfspace(test::column({1.0, 0.0}), 3.0))};
    ReachProblem zero = problem;
    zero.t_final = 0;
    check(test::raises([&] { sys.verify(zero, specs, alg::ReachAvoidSupportFunction{}); },
                       Error::wrong_value),
          "verify: t_final 0");
    check(test::raises([&] { sys.verify(problem, {}, alg::ReachAvoidSupportFunction{}); },
                       Error::wrong_value),
          "verify: no specification");
    const bool on = sys.verify(problem, specs, alg::ReachAvoidSupportFunction{}).verified;
    ChecksScope off(false);
    check(sys.verify(problem, specs, alg::ReachAvoidSupportFunction{}).verified == on,
          "verify: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const Zonotope R0 = zono({1, 0}, {0.1, 0, 0, 0.1}, 2);
    const Zonotope none = zono({0}, {}, 0);
    const Zonotope u05 = zono({0.05}, {0.05}, 1);
    const LinearSys osc0(mat(2, 2, Aosc), mat(2, 1, {0, 0}));
    const LinearSys osc1(mat(2, 2, Aosc), mat(2, 1, {0, 1}));
    const LinearSys nil(mat(2, 2, Anil), mat(2, 1, {0, 1}));
    const LinearSys sys3(mat(3, 3, A3), mat(3, 1, {0, 0, 1}), mat(1, 3, {1, 1, 0}));
    const Zonotope R03 = zono({1, 1, 1}, {0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1}, 3);
    const alg::ReachAvoidSupportFunction S{};
    const std::vector<double> none0;

    // Without an input set the horizon shrinks to what is not yet verified (iteration 3: 2275).
    matches("A", osc0.verify({R0, none, 5}, {safe({1, 0}, 1.3)}, S),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("J", osc0.verify({R0, none, 5}, {safe({0, 1}, 0.6993)}, S),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("J2", osc0.verify({R0, none, 5}, {safe({0, 1}, 0.69925)}, S),
            {true, 3, 0.0020000000000000005, 2275, -1, none0});
    matches("P", osc0.verify({R0, none, 5}, {safe({0, 1}, 0.7), safe({0, -1}, -0.7)}, S),
            {false, 0, 0, 0, 0, {1, -0.10000000000000001}});
    // Falsification: a hit of the unsafe set by the trajectory from the extreme point of R0.
    matches("C", osc0.verify({R0, none, 5}, {unsafe({1, 0}, -0.75)}, S),
            {false,
             1,
             0.050000000000000003,
             100,
             2.5500000000000003,
             {1.1000000000000001, -0.10000000000000001}});
    matches("M2", osc0.verify({R0, none, 5}, {unsafe({0, -1}, -0.65)}, S),
            {false,
             1,
             0.050000000000000003,
             100,
             4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("C2", osc0.verify({R0, none, 5}, {unsafe({1, 0}, -1.5)}, S),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    // With an input set (A invertible).
    matches("D", osc1.verify({R0, u05, 5}, {safe({1, 0}, 2)}, S),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    matches("K", osc1.verify({R0, u05, 5}, {safe({0, 1}, 0.7958)}, S),
            {true, 2, 0.010000000000000002, 500, -1, none0});
    matches("K2", osc1.verify({R0, u05, 5}, {safe({0, 1}, 0.7943)}, S),
            {true, 3, 0.0020000000000000005, 2500, -1, none0});
    matches("M", osc1.verify({R0, u05, 5}, {unsafe({0, -1}, -0.75)}, S),
            {false,
             1,
             0.050000000000000003,
             100,
             4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("N", osc1.verify({R0, u05, 5}, {unsafe({1, 0}, -1.0)}, S),
            {true, 1, 0.050000000000000003, 100, -1, none0});
    // expected results of the verification
    matches("Q", osc1.verify({R0, u05, 5}, {safe({1, 0}, 3), unsafe({0, -1}, -0.75)}, S),
            {false,
             1,
             0.050000000000000003,
             100,
             4.2000000000000002,
             {1.1000000000000001, -0.10000000000000001}});
    matches("E", osc1.verify({R0, u05, 5}, {unsafe({0, 1}, 1.9)}, S),
            {false, 0, 0, 0, 0, {1, -0.10000000000000001}});
    // Singular A (the series of the particular solution instead of A^-1).
    matches("O", nil.verify({R0, u05, 2}, {safe({0, 1}, 0.3005)}, S),
            {true, 1, 0.02, 100, -1, none0});
    matches("G2", nil.verify({R0, u05, 2}, {unsafe({1, 0}, 5)}, S),
            {false, 0, 0, 0, 0, {0.90000000000000002, 0}});
    // An output matrix C, a three-dimensional system, several specifications.
    matches("F",
            sys3.verify({R03, zono({0.1}, {0.05}, 1), 4}, {unsafe({1}, -3), unsafe({-1}, -3)}, S),
            {true, 1, 0.040000000000000001, 100, -1, none0});
    matches("L", sys3.verify({R03, zono({0.1}, {0.05}, 1), 4}, {safe({1}, 2.202)}, S),
            {true, 2, 0.0080000000000000002, 500, -1, none0});

    // Sparse A (40-chain): A = 5 (-2 I + sub + super), A(1,2) = 7, R0 = zonotope(e1, [.2 e1 .1 e2])
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
    const Zonotope zero_gen = zono({0}, {0}, 1), u1 = zono({0.5}, {0.1}, 1);
    matches("chain safe", chain0.verify({R0ch, zero_gen, 2}, {safe({1}, 0.014)}, S),
            {true, 1, 0.02, 100, -1, none0});
    matches("chain hit", chain0.verify({R0ch, zero_gen, 2}, {safe({1}, 0.012)}, S),
            {false, 1, 0.02, 100, 1.98, none0});
    matches("chain hit 2", chain0.verify({R0ch, zero_gen, 2}, {safe({1}, 0.011)}, S),
            {false, 1, 0.02, 100, 1.84, none0});
    matches("chain input", chain1.verify({R0ch, u1, 2}, {safe({1}, 0.005)}, S),
            {false, 1, 0.02, 100, 1.16, none0});
    // the chain with the second halfspace
    matches("chain input 2", chain1.verify({R0ch, u1, 2}, {safe({1}, 0.001)}, S),
            {false, 1, 0.02, 100, 0.72, none0});

    // Unsupported input is refused; the support function algorithm takes one halfspace only.
    check(test::throws([&] {
              osc1.verify({R0, u05, 5},
                          {Specification::unsafe_set(test::polytope(
                              {{test::column({1, 0}), 1.0}, {test::column({0, 1}), 1.0}}))},
                          S);
          }),
          "an unsafe set of several halfspaces is refused by the support function algorithm");
    check(test::throws([&] { osc1.verify({R0, u05, 5}, {}, S); }), "no specification");
    check(test::throws([&] { osc1.verify({R0, u05, 5}, {safe({1}, 2)}, S); }), "wrong normal size");
    check(test::throws([&] { osc1.verify({R0, u05, 0}, {safe({1, 0}, 2)}, S); }), "t_final 0");
    check(
        test::throws([&] { osc1.verify({R0, zono({0, 0}, {1, 0}, 1), 5}, {safe({1, 0}, 2)}, S); }),
        "U of the wrong dimension");
    checks_and_their_switch();
    return test::finish("linear_sys verify");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
