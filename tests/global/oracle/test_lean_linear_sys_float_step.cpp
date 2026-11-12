// test_lean_linear_sys_float_step - the linear step of the nonlinear algorithms against CORALean
//
// One step of x' = A x + u, u in U, from X0 is CORALean's reachFloatInput for one step (the
// oracle's linearSys.reach with B = I): the window H(tau) + P^R and the set at the end of the step
// agree bit for bit at nearest, for a random system and for a stiff one that raises the Taylor
// order. The step follows the interval-matrix exponential, the correction matrix and the input
// solutions of CORALean's float layer.
//
// ORACLE: contSet/@zonotope/mtimes.m
// CORA.cpp only: CORA's oneStep is a different step (the input curvature in the window)
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_step.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// One step of the system A from X0 under U, CORA.cpp's against the oracle's.
void aux_compare(const std::string &name, const Tensor &A, const Zonotope &X0, const Zonotope &U,
                 double dt, int eta, int rho) {
    const int64_t n = A.size(-1);
    const Tensor zero = A.zeros_like({n, 1});
    const std::optional<FloatStep> mine = priv_float_one_step(A, zero, U, X0, dt, eta, rho);
    const lean::ReachSet theirs =
        lean::LinearSys(A, A.eye_like())
            .reach(lean::Zonotope(X0), lean::Zonotope(U), dt,
                   alg::Standard{.time_step = dt, .taylor_terms = eta, .zonotope_order = rho});
    test::check(mine.has_value(), name + ": an admissible order");
    if (!mine) return;
    test::oracle::check(name + " window", mine->first, theirs.time_interval[0]);
    test::oracle::check(name + " end", mine->second, theirs.time_point[1]);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.reach", [](test::oracle::Random &rnd) {
        const Tensor A = rnd.matrix(3, 3);
        for (const int gx : {2, 4})
            for (const int gu : {2, 3})
                for (const int eta : {4, 6})
                    for (const int rho : {1, 10})
                        aux_compare("gx" + std::to_string(gx) + " gu" + std::to_string(gu) +
                                        " eta" + std::to_string(eta) + " rho" + std::to_string(rho),
                                    A, rnd.zonotope(3, gx), rnd.zonotope(3, gu), 0.05, eta, rho);
    });
    // the map by an interval matrix that every term of the step is made of
    test::oracle::run("zonotope.mtimesInterval", [](test::oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 5}) {
            const Zonotope Z = rnd.zonotope(n, 4);
            const Interval M = rnd.box_matrix(n, n);
            test::oracle::check(Z.linear_map(M), lean::Zonotope(Z).linear_map(lean::Interval(M)));
        }
    });
    return test::finish("lean linearSys float step");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
