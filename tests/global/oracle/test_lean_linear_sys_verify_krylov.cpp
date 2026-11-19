// test_lean_linear_sys_verify_krylov - verify's sound Krylov algorithm against CORALean
//
// The Krylov algorithm keeps CORA.cpp's tuning (the plan of each direction's restart windows, the
// halving of the step, the Lyapunov weight it tries) and decides by CORALean's
// LinearSys.verifyKrylovPassFloat: the windows of every direction from the ball of e^{H L} with
// the start's error, their Gershgorin carry and the smaller weighted one, and the support bounds
// of every step from the iterates of the ball of e^{H dt} with their carried error. Here against
// the oracle, bit for bit at nearest, on systems of two to six states (a safe set that holds, one
// that does not, an unsafe set, inputs, an output matrix, a non-normal matrix whose weight is in
// use): the verdict of the last pass, every direction's windows (order, nu, error) and bounds.
//
// ORACLE: contDynamics/@linearSys/verify.m
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cont_dynamics/linear_sys/private/priv_verify_krylov.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"
#include "spec_testing.h"

#include <string>
#include <utility>
#include <vector>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A problem: x' = A x + B u, y = C x, R0, U and the specifications.
struct Problem {
    Tensor A, B, C;
    Zonotope R0, U;
    std::vector<Specification> specs;
    double t_final = 1.0, time_step = 0.01, krylov_error = numerics::kKrylovError;
};

/// A box about c of radius r in every coordinate.
Zonotope aux_box(const std::vector<double> &c, double r) {
    const auto n = int64_t(c.size());
    std::vector<double> G(std::size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) G[std::size_t(i * n + i)] = r;
    return {Tensor::from_data(c, {n}), Tensor::from_data(G, {n, n})};
}

/// The halfspace a'y <= b as a safe or an unsafe set.
Specification aux_half(const std::vector<double> &a, double b, bool safe) {
    const Polytope H = test::polytope({{Tensor::from_data(a, {int64_t(a.size()), 1}), b}});
    return safe ? Specification::safe_set(H) : Specification::unsafe_set(H);
}

/// verify by the Krylov algorithm, its last pass against the oracle's.
void aux_verify(const std::string &name, const Problem &P) {
    VerifyKrylovTrace trace;
    const LinearSys sys(P.A, P.B, P.C);
    alg::ReachAvoidKrylov alg;
    alg.time_step = P.time_step;
    alg.krylov_error = P.krylov_error;
    alg.krylov_step = 2;
    const VerifyResult mine =
        priv_verify_ra_krylov(sys, P.R0, P.U, P.t_final, P.specs, alg, &trace);
    test::check(mine.verified == trace.verified, name + ": the trace is of the last pass");
    // the oracle's pass on the same directions, plans and weight
    const int64_t n = P.A.shape()[0];
    const CanonicalInput input = priv_canonical_input(sys, P.U);
    std::vector<lean::VerifyKrylovDirection> dirs;
    dirs.reserve(trace.dirs.size());
    for (const VerifyKrylovTraceDirection &d : trace.dirs)
        dirs.push_back({Tensor::from_data(d.w, {n, 1}), d.plan, d.limit, d.strict, d.group});
    std::optional<lean::VerifyWeight> cert;
    if (!trace.P.empty())
        cert = lean::VerifyWeight{Tensor::from_data(trace.P, {n, n}),
                                  Tensor::from_data(trace.L, {n, n}), trace.a};
    const lean::LinearSys oracle(P.A);
    const lean::VerifyKrylovPass theirs = oracle.verify_krylov_pass(
        P.R0.center().expand_dims(-1), P.R0.generators().matrix_transpose(), input.u_trans,
        input.GU, dirs, trace.groups, trace.dt0, trace.halvings, trace.steps, machine_eps(), cert);
    test::oracle::check(name + " verified", trace.verified, theirs.verified);
    test::check(theirs.dirs.size() == trace.dirs.size(), name + ": the same directions");
    // each direction: its windows, then its bounds, bit for bit
    for (std::size_t i = 0; i < trace.dirs.size() && i < theirs.dirs.size(); ++i) {
        const std::string at = name + " direction " + std::to_string(i);
        const VerifyKrylovTraceDirection &a = trace.dirs[i];
        const lean::VerifyKrylovDirectionResult &b = theirs.dirs[i];
        test::check(b.windows.has_value() || a.windows.empty(), at + ": windows computed");
        if (b.windows) {
            test::check(b.windows->size() == a.windows.size(), at + ": the same windows");
            for (std::size_t j = 0; j < a.windows.size() && j < b.windows->size(); ++j) {
                const std::string w = at + " window " + std::to_string(j);
                test::check(a.windows[j].order == (*b.windows)[j].order, w + ": the same order");
                test::oracle::check(w + " nu", a.windows[j].norm, (*b.windows)[j].nu);
                test::oracle::check(w + " err", a.windows[j].error, (*b.windows)[j].err);
            }
        }
        // the bounds on every step
        test::check(a.bounds.has_value() == b.bounds.has_value(), at + ": bounds in both");
        if (!a.bounds || !b.bounds) continue;
        test::check(a.bounds->size() == b.bounds->size(), at + ": the same steps");
        for (std::size_t j = 0; j < a.bounds->size() && j < b.bounds->size(); ++j)
            test::oracle::check(at + " bound " + std::to_string(j), (*a.bounds)[j], (*b.bounds)[j]);
    }
}

/// The damped rotation x' = [-0.5 2; -2 -0.5] x + [1; 0.5] u, y = x, from a box around (1, 0)
/// for one second, against the specification at the step size.
Problem aux_rotation(const Specification &spec, double time_step) {
    const Zonotope U(Tensor::from_data({0.2}, {1}), Tensor::from_data({0.1}, {1, 1}));
    return {Tensor({{-0.5, 2.0}, {-2.0, -0.5}}),
            Tensor({{1.0}, {0.5}}),
            Tensor::eye(2),
            aux_box({1.0, 0.0}, 0.1),
            U,
            {spec},
            1.0,
            time_step};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.verifyKrylovPass", [](test::oracle::Random &rnd) {
        // a damped rotation, y = x: a safe halfspace that holds and one the start leaves
        const Zonotope U1(Tensor::from_data({0.2}, {1}), Tensor::from_data({0.1}, {1, 1}));
        aux_verify("holds", aux_rotation(aux_half({1.0, 0.0}, 1.5, true), 0.02));
        aux_verify("fails", aux_rotation(aux_half({1.0, 0.0}, 0.95, true), 0.02));
        aux_verify("unsafe", aux_rotation(aux_half({-1.0, 0.0}, -1.6, false), 0.02));
        // a limit that a finer step reaches: the step is halved
        for (const double b : {1.25, 1.22})
            aux_verify("halved " + std::to_string(b),
                       aux_rotation(aux_half({1.0, 0.0}, b, true), 0.25));
        // a stable but non-normal block: the Gershgorin carry grows, the weight is tried
        const Tensor A3({{-1.0, 8.0, 0.0}, {0.0, -1.0, 8.0}, {0.0, 0.0, -1.0}});
        const Tensor B3({{0.0}, {0.0}, {1.0}});
        const Tensor C3({{1.0, 0.0, 0.0}});
        aux_verify("non-normal", {A3,
                                  B3,
                                  C3,
                                  aux_box({0.1, 0.0, 0.1}, 0.01),
                                  U1,
                                  {aux_half({1.0}, 50.0, true)},
                                  2.0,
                                  0.04});
        // six states, two inputs, two outputs, random stable dynamics
        const Tensor A6 = rnd.matrix(6, 6) - Tensor::eye(6) * 3.0;
        const Zonotope U2(Tensor::from_data({0.1, -0.1}, {2}), Tensor::eye(2) * 0.05);
        aux_verify("six states",
                   {A6,
                    rnd.matrix(6, 2),
                    rnd.matrix(2, 6),
                    aux_box({1.0, 0.5, 0.0, -0.5, 0.2, 0.1}, 0.05),
                    U2,
                    {aux_half({1.0, 0.0}, 3.0, true), aux_half({0.0, 1.0}, -3.0, false)},
                    1.0,
                    0.02});
        // twenty states and a tight tolerance: restart windows; a limit a finer step reaches
        const Tensor A20 = rnd.matrix(20, 20) * 0.5 - Tensor::eye(20) * 2.0;
        std::vector<double> c20(20, 0.1);
        std::vector<double> l20(20, 0.0);
        l20[0] = 1.0;
        Problem W{A20,
                  rnd.matrix(20, 1),
                  Tensor::eye(20),
                  aux_box(c20, 0.01),
                  U1,
                  {aux_half(l20, 0.3, true)},
                  1.0,
                  0.05,
                  1e-14};
        aux_verify("windows", W);
    });
    return test::finish("lean linearSys verify krylov");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
