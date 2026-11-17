// test_lean_linear_sys_verify_support - verify's sound support function algorithm against CORALean
//
// A homogeneous system whose balanced logarithmic norm bounds the growth of e^{tA} (the heat
// equation's case) runs CORALean's LinearSys.verifySupportFuncFloat: the exponential as a matrix
// and a balanced radius (MatExp.expBallFloat), the directions with their error carried by the
// growth bound, CORA's distances from above and below, and the exact unverified times. Here
// against the oracle, bit for bit at nearest: e^{A dt} and its radius on random stable systems and
// a stiff one, and the verdict, the passes, the last step size and count and the falsifying time
// point of verify on heat-like chains (verified in one to four passes, falsified, a safe and an
// unsafe specification, an output matrix, a small growth). HEAT01 itself is left out:
// the oracle takes over half an hour for its 125 states.
//
// ORACLE: contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/tensor/tolerance.h"
#include "oracle_testing.h"
#include "spec_testing.h"

#include <string>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kFuel = 1000; // passes the oracle may run

/// e^{A dt} and its radius, CORA.cpp's against the oracle's.
void aux_ball(const std::string &name, const Tensor &A, double dt) {
    const int64_t n = A.shape()[0];
    const std::vector<double> s = priv_verify_balance_scales(A.data(), n);
    const std::optional<VerifyExpBall> mine = priv_verify_exp_ball(A, s, dt);
    const std::optional<lean::VerifyExpBall> theirs = lean::LinearSys(A).exp_ball(dt);
    test::check(mine.has_value() == theirs.has_value(), name + ": both or neither");
    if (!mine || !theirs) return;
    test::oracle::check(name + " D", mine->D, theirs->D);
    test::oracle::check(name + " r", mine->r, theirs->r);
}

/// A row of the halfspaces C_s y <= d as verify builds it: a safe a'y <= b as it is, an unsafe one
/// as -a'y <= -b.
struct Row {
    std::vector<double> a;
    double b;
    bool safe;
};

/// verify of x' = A x (y = C x) from R0 against the rows, CORA.cpp's against the oracle's.
void aux_verify(const std::string &name, const Tensor &A, const std::optional<Tensor> &C,
                const Zonotope &R0, const std::vector<Row> &rows, double t_final) {
    const int64_t n = A.shape()[0];
    const int64_t p = C ? C->shape()[0] : n;
    std::vector<Specification> specs;
    std::vector<double> Cs, d;
    for (const Row &r : rows) {
        const Polytope H = test::halfspace(test::column(r.a), r.b);
        specs.push_back(r.safe ? Specification::safe_set(H) : Specification::unsafe_set(H));
        for (double v : r.a) Cs.push_back(r.safe ? v : -v);
        d.push_back(r.safe ? r.b : -r.b);
    }
    // CORA.cpp's verify without input, and the oracle's run on the same rows
    const auto S = static_cast<int64_t>(rows.size());
    const LinearSys sys(A, A.zeros_like({n, 1}), C);
    const Zonotope none(A.zeros_like({1, 1}), A.zeros_like({1, 0}));
    const VerifyResult mine =
        sys.verify({R0, none, t_final}, specs, alg::ReachAvoidSupportFunction{});
    const lean::VerifySupportRun theirs =
        lean::LinearSys(A, std::nullopt, C)
            .verify_support_func(R0.center(), R0.generators(), Tensor::from_data(Cs, {S, p}),
                                 Tensor::from_data(d, {S, 1}), t_final, numerics::kMinTimeStep,
                                 kFuel);
    test::oracle::check(name + " verified", mine.verified, theirs.verified);
    test::oracle::check(name + " falsified", mine.falsification.has_value(), theirs.falsified);
    test::oracle::check(name + " iterations", int64_t(mine.iterations), theirs.iterations);
    test::oracle::check(name + " steps", int64_t(mine.num_steps), theirs.num_steps);
    test::oracle::check(name + " step size", mine.time_step, theirs.time_step);
    if (mine.falsification && theirs.fals_step)
        test::oracle::check(name + " falsifying time", mine.falsification->time,
                            double(*theirs.fals_step) * theirs.time_step);
}

/// The heat equation on a chain of n cells: -2 on the diagonal, 1 beside it, the ends insulated.
Tensor aux_chain(int64_t n, double k) {
    std::vector<double> a(std::size_t(n) * n, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        const double left = i > 0 ? k : 0, right = i + 1 < n ? k : 0;
        a[std::size_t(i) * n + i] = -(left + right) - (i == 0 ? k : 0);
        if (i > 0) a[std::size_t(i) * n + i - 1] = k;
        if (i + 1 < n) a[std::size_t(i) * n + i + 1] = k;
    }
    return Tensor::from_data(a, {n, n});
}

/// A box about c of radius r in every coordinate.
Zonotope aux_box(const std::vector<double> &c, double r) {
    const auto n = static_cast<int64_t>(c.size());
    std::vector<double> G(std::size_t(n) * n, 0.0);
    for (int64_t i = 0; i < n; ++i) G[std::size_t(i) * n + i] = r;
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, n})};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.expBall", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 5})
            for (const double dt : {0.01, 0.25, 2.0})
                aux_ball("n" + std::to_string(n) + " dt" + std::to_string(dt),
                         rnd.matrix(n, n) - Tensor::eye(n) * 2.0, dt);
        aux_ball("chain", aux_chain(6, 3.0), 0.4);
        aux_ball("stiff", Tensor({{-1000.0, 1.0}, {0.0, -0.5}}), 0.125);
    });
    test::oracle::run("linearSys.verifySupportFunc", [](test::oracle::Random &) {
        const Tensor A = aux_chain(4, 1.0);
        const Zonotope R0 = aux_box({1.0, 0.5, 0.0, 0.0}, 0.05);
        const std::vector<double> last{0, 0, 0, 1};
        // the last cell peaks at 0.22034 near t = 2.8: the closer the bound, the more passes
        aux_verify("one pass", A, std::nullopt, R0, {{last, 0.221, true}}, 4.0);
        aux_verify("two passes", A, std::nullopt, R0, {{last, 0.2205, true}}, 4.0);
        aux_verify("three passes", A, std::nullopt, R0, {{last, 0.220345, true}}, 4.0);
        aux_verify("four passes", A, std::nullopt, R0, {{last, 0.220336, true}}, 4.0);
        aux_verify("falsified", A, std::nullopt, R0, {{last, 0.2203, true}}, 4.0);
        // the touching start never verifies; the step shrinks below 1e-12
        aux_verify("touching", A, std::nullopt, R0, {{{1, 0, 0, 0}, 1.05, true}}, 2.0);
        aux_verify("unsafe at the start", A, std::nullopt, R0, {{{-1, 0, 0, 0}, -0.3, false}}, 3.0);
        aux_verify("unsafe later", A, std::nullopt, R0, {{{-1, 0, 0, 0}, -0.4, false}}, 3.0);
        aux_verify("two rows", A, std::nullopt, R0,
                   {{{1, 0, 0, 0}, 1.2, true}, {last, 0.2205, true}}, 4.0);
        aux_verify("output", A, Tensor({{0.5, 0.5, 0.0, 0.0}}), R0, {{{1}, 0.85, true}}, 3.0);
        // a small growth (mu > 0, so a growth bound above 1)
        aux_verify("growing", A + Tensor::eye(4) * 0.05, std::nullopt, R0, {{last, 0.26, true}},
                   4.0);
    });
    return test::finish("lean linearSys verify support");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
