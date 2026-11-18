// test_lean_linear_sys_verify_support - verify's sound support function algorithm against CORALean
//
// A system whose balanced logarithmic norm or energy weight bounds the growth of e^{tA} runs
// CORALean's LinearSys.verifySupportFuncFloat: the exponential as a matrix and a balanced radius,
// the directions with their error carried by the growth bound, CORA's distances from above and
// below with the input's terms, and the exact unverified times. Here against the oracle, bit for
// bit at nearest: e^{A dt} and its radius on random stable systems and a stiff one; the verdict,
// the passes, the last step size and count and the falsifying time of verify on heat-like chains
// (one to four passes, falsified, safe and unsafe rows, an output matrix, a small growth); the
// energy weight of beams (an undamped one the check rejects, matrices off the form); verify on a
// beam, and with inputs: a constant and a varying force, each alone, two inputs through a dense B
// with a band of an output (falsified in the third pass), a forced beam. HEAT01 and the ARCH
// beams are left out: the oracle takes over half an hour for HEAT01's 125 states.
//
// ORACLE: contDynamics/@linearSys/verify.m
// ORACLE: contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_verify_energy.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/tensor/tolerance.h"
#include "oracle_testing.h"
#include "spec_testing.h"

#include <algorithm>
#include <cmath>
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

/// verify of x' = A x + B u, u in U (y = C x) from R0 against the rows, CORA.cpp's against the
/// oracle's; no input without B.
void aux_verify(const std::string &name, const Tensor &A, const std::optional<Tensor> &C,
                const Zonotope &R0, const std::vector<Row> &rows, double t_final,
                const std::optional<Tensor> &B = std::nullopt,
                const std::optional<Zonotope> &U = std::nullopt) {
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
    // CORA.cpp's verify, and the oracle's run on the same rows
    const auto S = static_cast<int64_t>(rows.size());
    const LinearSys sys(A, B ? *B : A.zeros_like({n, 1}), C);
    const Zonotope none(A.zeros_like({1, 1}), A.zeros_like({1, 0}));
    const VerifyResult mine =
        sys.verify({R0, B ? *U : none, t_final}, specs, alg::ReachAvoidSupportFunction{});
    const lean::VerifySupportRun theirs = lean::LinearSys(A, B, C).verify_support_func(
        R0.center(), R0.generators(), Tensor::from_data(Cs, {S, p}), Tensor::from_data(d, {S, 1}),
        t_final, numerics::kMinTimeStep, kFuel, U);
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

/// A beam of h masses m_i on springs k (fixed ends), damping alpha + beta K, and z inputs pushing
/// the last mass: [[0, I, 0], [-K, -C, F], [0, 0, 0]] with K = M^-1 K_s.
Tensor aux_beam(int64_t h, double k, const std::vector<double> &m, double alpha, double beta,
                int64_t z) {
    const int64_t n = 2 * h + z;
    std::vector<double> a(std::size_t(n) * n, 0.0);
    auto at = [&](int64_t i, int64_t j) -> double & { return a[std::size_t(i * n + j)]; };
    for (int64_t i = 0; i < h; ++i) {
        at(i, h + i) = 1.0;
        const double mi = m[std::size_t(i) % m.size()];
        for (int64_t j = std::max<int64_t>(0, i - 1); j <= std::min(h - 1, i + 1); ++j) {
            const double kij = (i == j ? 2.0 * k : -k) / mi;
            at(h + i, j) = -kij;
            at(h + i, h + j) = -(i == j ? alpha + beta * kij : beta * kij);
        }
    }
    for (int64_t q = 0; q < z; ++q) at(2 * h - 1 - q, 2 * h + q) = 1.0 / double(q + 1);
    return Tensor::from_data(a, {n, n});
}

/// The energy weight and its growth factor, CORA.cpp's against the oracle's.
void aux_energy(const std::string &name, const Tensor &A, double t_final) {
    const int64_t n = A.shape()[0];
    const std::vector<double> a = A.data();
    const std::optional<EnergyWeight> w = priv_verify_energy_weight(a, n, 1.0 / t_final);
    const lean::VerifyEnergyGrowth theirs = lean::LinearSys(A).energy_growth(t_final);
    test::check(w.has_value() == theirs.P.has_value(), name + ": a weight on both or neither");
    if (w && theirs.P) {
        std::vector<double> P(std::size_t(n) * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (const auto &[j, v] : w->P[std::size_t(i)]) P[std::size_t(i * n + j)] = v;
        test::oracle::check(name + " P", Tensor::from_data(P, {n, n}), *theirs.P);
    }
    const std::vector<double> s = priv_verify_balance_scales(a, n);
    const std::optional<double> kappa = priv_verify_energy_growth(a, s, n, 1.0 / t_final);
    test::check(kappa.has_value() == theirs.kappa.has_value(),
                name + ": accepted by both or neither");
    if (kappa && theirs.kappa) test::oracle::check(name + " kappa", *kappa, *theirs.kappa);
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
    test::oracle::run("linearSys.energyGrowth", [](test::oracle::Random &) {
        const std::vector<double> m{1.0, 2.0, 1.0, 3.0};
        aux_energy("beam 3", aux_beam(3, 100.0, m, 0.1, 0.01, 1), 1.0);
        aux_energy("beam 4 no input", aux_beam(4, 50.0, m, 0.05, 0.02, 0), 2.0);
        aux_energy("beam 2 two inputs", aux_beam(2, 1e4, {1.0}, 1.0, 1e-3, 2), 0.01);
        aux_energy("undamped", aux_beam(3, 10.0, m, 0.0, 0.0, 1), 1.0); // rate check fails
        // one ulp off the form, and no positions at all: no weight
        Tensor off = aux_beam(3, 100.0, m, 0.1, 0.01, 1);
        std::vector<double> a = off.data();
        a[3] = std::nextafter(1.0, 2.0);
        aux_energy("one ulp off", Tensor::from_data(a, {7, 7}), 1.0);
        aux_energy("heat", aux_chain(4, 1.0), 1.0);
    });
    test::oracle::run("linearSys.verifySupportFunc", [](test::oracle::Random &) {
        // a beam grows in no log norm; its energy weight bounds the growth
        const Tensor A = aux_beam(3, 100.0, {1.0, 2.0, 1.0}, 0.1, 0.01, 1);
        std::vector<double> c(7, 0.0);
        c[6] = 1.0;
        std::vector<double> G(7, 0.0);
        G[0] = 0.01;
        const Zonotope R0(Tensor::from_data(c, {7, 1}), Tensor::from_data(G, {7, 1}));
        const std::vector<double> mid{0, 0, 1, 0, 0, 0, 0};
        aux_verify("beam verified", A, std::nullopt, R0, {{mid, 0.2, true}}, 1.0);
        aux_verify("beam falsified", A, std::nullopt, R0, {{mid, 0.005, true}}, 1.0);
    });
    test::oracle::run("linearSys.verifySupportFunc", [](test::oracle::Random &) {
        // two masses pushed by a constant and a varying force on the last velocity (CBF's form)
        const Tensor A({{0.0, 0.0, 1.0, 0.0},
                        {0.0, 0.0, 0.0, 1.0},
                        {-2.0, 1.0, -0.1, 0.0},
                        {1.0, -2.0, 0.0, -0.1}});
        const Tensor I = Tensor::eye(4);
        const Zonotope R0(Tensor::from_data({0, 0, 0, 0}, {4, 1}), A.zeros_like({4, 0}));
        const Zonotope U(Tensor::from_data({0, 0, 0, 0.5}, {4, 1}),
                         Tensor::from_data({0, 0, 0, 0.3}, {4, 1}));
        const std::vector<double> first{1, 0, 0, 0};
        aux_verify("forced verified", A, std::nullopt, R0, {{first, 0.05, true}}, 1.0, I, U);
        aux_verify("forced falsified", A, std::nullopt, R0, {{first, 0.01, true}}, 1.0, I, U);
        // the peak lies at the horizon: falsified in the fourth pass
        aux_verify("forced four passes", A, std::nullopt, R0, {{first, 0.028, true}}, 1.0, I, U);
        // a constant force only, a varying one only
        const Zonotope Uc(U.center(), A.zeros_like({4, 0}));
        aux_verify("constant force", A, std::nullopt, R0, {{first, 0.05, true}}, 1.0, I, Uc);
        const Zonotope Uv(A.zeros_like({4, 1}), U.generators());
        aux_verify("varying force", A, std::nullopt, R0, {{first, 0.05, true}}, 1.0, I, Uv);
        // two inputs through a dense B, an initial box, a safe band of an output (ISS's form)
        const Tensor B({{0.0, 0.0}, {0.0, 0.0}, {0.5, -0.25}, {0.125, 0.75}});
        const Zonotope U2(Tensor::from_data({0.1, -0.1}, {2, 1}),
                          Tensor({{0.05, 0.0}, {0.0, 0.05}}));
        const Zonotope X0 = aux_box({0.01, 0, 0, 0}, 0.01);
        const Tensor C({{0.0, 1.0, 0.0, 0.0}});
        aux_verify("band verified", A, C, X0, {{{1}, 0.2, true}, {{-1}, 0.2, true}}, 1.0, B, U2);
        aux_verify("band three passes", A, C, X0, {{{1}, 0.0515, true}, {{-1}, 0.0515, true}}, 1.0,
                   B, U2);
        aux_verify("band falsified", A, C, X0, {{{1}, 0.05, true}, {{-1}, 0.05, true}}, 1.0, B, U2);
        // a beam of three masses with its force through B
        const Tensor Ab = aux_beam(3, 100.0, {1.0, 2.0, 1.0}, 0.1, 0.01, 0);
        std::vector<double> b(6, 0.0);
        b[5] = 1.0;
        const Zonotope Ub(Tensor::from_data({0.4}, {1, 1}), Tensor::from_data({0.2}, {1, 1}));
        const Zonotope Rb(A.zeros_like({6, 1}), A.zeros_like({6, 0}));
        const std::vector<double> mid{0, 1, 0, 0, 0, 0};
        aux_verify("beam forced verified", Ab, std::nullopt, Rb, {{mid, 0.02, true}}, 1.0,
                   Tensor::from_data(b, {6, 1}), Ub);
        aux_verify("beam forced falsified", Ab, std::nullopt, Rb, {{mid, 0.001, true}}, 1.0,
                   Tensor::from_data(b, {6, 1}), Ub);
    });
    return test::finish("lean linearSys verify support");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
