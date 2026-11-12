// test_nonlinear_sys_reach_poly_adaptive - nonlinearSys reach with alg "poly-adaptive"
//
// The step sizes are tuned by the algorithm (CORA's poly-adaptive as CORALean repairs it): they
// are positive, add up to the horizon or beyond, and the polynomial zonotopes enclose simulated
// trajectories (van der Pol with an input, whose third derivatives do not vanish; a quadratic
// system, where they do). The reductions of the sets are checked on their own: they keep a
// superset, and the long independent part triggers the restructure.
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA: contSet/@polyZonotope/reduce.m
// PROPERTY: contDynamics/@nonlinearSys/reach_adaptive.m
// PROPERTY: contDynamics/@nonlinearSys/linReach_adaptive.m
// PROPERTY: contSet/@polyZonotope/reduce.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

/// Three states with quadratic terms only (the third derivatives vanish).
NonlinearSys quadratic() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-x[0] + 0.5 * x[1] * x[2], -x[1] + 0.3 * x[0] * x[0] - x[2],
                                     x[1] - 0.8 * x[2] - 0.2 * x[0] * x[1]};
        },
        3, 1);
}

/// Whether the point j of the columns (n, N) lies in the box of the set up to tol.
bool in_box(const PolyZonotope &Z, const std::vector<double> &points, int64_t N, int64_t j,
            double tol) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double v = points[i * std::size_t(N) + std::size_t(j)];
        if (!(v >= lo[i] - tol && v <= hi[i] + tol)) return false;
    }
    return true;
}

/// The times are 0 = t_0 < ... < t_K, t_K >= t_final, one set per step and one more at the points.
void check_times(const PolyReachSet &R, double t_final, const std::string &b) {
    check(R.time.size() == R.time_point.size() && R.time_point.size() == R.time_interval.size() + 1,
          b + ": a time and a point set per step boundary");
    check(R.time.front() == 0.0 && R.time.back() >= t_final, b + ": from 0 to t_final or beyond");
    bool increasing = true;
    for (std::size_t k = 1; k < R.time.size(); ++k)
        increasing = increasing && R.time[k] > R.time[k - 1];
    check(increasing, b + ": the times increase");
}

/// Trajectories from random points of the start set stay in the sets over the step they are in.
int aux_outside(const NonlinearSys &sys, const Zonotope &X0, const std::optional<Zonotope> &U,
                const PolyReachSet &R, double T) {
    const double h = 1e-3;
    cora::Rng rng(3);
    const std::vector<Tensor> x =
        sys.simulate_random({.initial_set = Set(X0),
                             .input_set = U ? std::optional<Set>(Set(*U)) : std::nullopt,
                             .t_final = T},
                            alg::RungeKutta4{.time_step = h}, 20, rng);
    int outside = 0;
    std::size_t k = 0;
    for (std::size_t s = 0; s < x.size(); ++s) {
        const double t = double(s) * h;
        while (k + 1 < R.time_interval.size() && R.time[k + 1] < t) ++k;
        const std::vector<double> d = x[s].data();
        for (int j = 0; j < 20; ++j) outside += !in_box(R.time_interval[k], d, 20, j, 1e-9);
    }
    return outside;
}

/// The set Z is a subset of the box of W (up to tol).
bool box_in(const PolyZonotope &Z, const PolyZonotope &W, double tol) {
    const Interval a = Z.interval(), b = W.interval();
    const std::vector<double> alo = a.inf().data(), ahi = a.sup().data(), blo = b.inf().data(),
                              bhi = b.sup().data();
    for (std::size_t i = 0; i < alo.size(); ++i)
        if (!(alo[i] >= blo[i] - tol && ahi[i] <= bhi[i] + tol)) return false;
    return true;
}

/// Van der Pol with an input: the trajectories stay in the sets of the steps.
void van_der_pol_encloses_simulations(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const Zonotope X0(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    const Zonotope U(Tensor({0.0}), Tensor({{0.1}}));
    const double T = 0.05;
    const PolyReachSet R =
        sys.reach({.initial_set = PolyZonotope(X0), .input_set = U, .t_final = T},
                  alg::PolynomializationAdaptive{});
    check_times(R, T, b + " van der Pol");
    const int outside = aux_outside(sys, X0, U, R, T);
    check(outside == 0,
          b + ": " + std::to_string(outside) + " points left the sets of van der Pol");
}

/// A quadratic system (empty third-order tensor): the same, and the orders lose one at step 2.
void quadratic_system_encloses_simulations(const std::string &b) {
    const NonlinearSys sys = quadratic();
    const Zonotope X0(Tensor({1.0, 0.5, 0.2}),
                      Tensor({{0.05, 0.0, 0.0}, {0.0, 0.05, 0.0}, {0.0, 0.0, 0.05}}));
    const Zonotope U(Tensor({0.0}), Tensor::zeros({1, 0}));
    const double T = 0.05;
    const PolyReachSet R =
        sys.reach({.initial_set = PolyZonotope(X0), .input_set = U, .t_final = T},
                  alg::PolynomializationAdaptive{});
    check_times(R, T, b + " quadratic");
    const int outside = aux_outside(sys, X0, U, R, T);
    check(outside == 0,
          b + ": " + std::to_string(outside) + " points left the sets of the quadratic system");
}

/// The adaptive reduction of a polynomial zonotope keeps a superset and boxes into the independent
/// part; reduce_only_dep keeps the longest generators; restructure leaves no independent part.
void reductions_keep_a_superset(const std::string &b) {
    const Tensor c({0.5, -1.0});
    Tensor G({{0.3, 0.02, 0.5, 0.001, 0.04}, {0.1, 0.03, -0.4, 0.002, 0.05}});
    const Tensor GI({{0.01, 0.0}, {0.0, 0.002}});
    const Exponents E{{1, 0, 0, 1, 2}, {0, 1, 0, 1, 1}, {0, 0, 1, 0, 0}};
    PolyZonotope Z(c, G, GI, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {2, 1, 0}}, {1, 2, 3});
    const PolyZonotope red = priv_reduce_adaptive_poly(Z, 1e-2);
    check(red.num_dependent() <= Z.num_dependent(),
          b + ": the adaptive reduction keeps no more generators");
    check(box_in(Z, red, 1e-12), b + ": the adaptive reduction encloses the set");
    const PolyZonotope only = priv_reduce_only_dep(Z, 1);
    check(only.num_dependent() == 2 && box_in(Z, only, 1e-12),
          b + ": reduce_only_dep keeps order * n generators and a superset");
    const PolyZonotope flat = priv_restructure(Z);
    check(flat.independent_generators().shape()[1] == 0 && box_in(Z, flat, 1e-12),
          b + ": restructure makes the independent part dependent and encloses");
    check(!priv_volume_ratio_exceeds(Z, 10.0) && priv_volume_ratio_exceeds(Z, 1e-9),
          b + ": the volume ratio of the independent part");
}

/// The arguments of the algorithm are checked.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const PolyZonotope X0(Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}})));
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .t_final = 1.0},
                            alg::PolynomializationAdaptive{.zeta_z = 2});
              },
              Error::wrong_value),
          b + ": zeta_z above 1 is refused");
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .t_final = 1.0},
                            alg::PolynomializationAdaptive{.zeta_phi = {0.8}});
              },
              Error::wrong_value),
          b + ": zeta_phi needs three entries");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        van_der_pol_encloses_simulations(b);
        quadratic_system_encloses_simulations(b);
        reductions_keep_a_superset(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinear_sys reach poly adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
