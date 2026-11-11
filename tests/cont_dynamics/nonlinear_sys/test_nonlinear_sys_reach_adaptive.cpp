// test_nonlinear_sys_reach_adaptive - nonlinearSys reach with alg "lin-adaptive"
//
// The step sizes are tuned by the algorithm (CORA's lin-adaptive as CORALean repairs it): they are
// positive, add up to the horizon or beyond, and the sets enclose simulated trajectories (van der
// Pol, with an input; the Robertson reaction, whose sum of concentrations stays a point in the
// limit); the formulas of the tuning are checked on their own.
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// PROPERTY: contDynamics/@nonlinearSys/reach_adaptive.m
// PROPERTY: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
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

NonlinearSys robertson() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-0.4 * x[0] + 100 * x[1] * x[2],
                                     0.4 * x[0] - 100 * x[1] * x[2] - 1000 * pow(x[1], 2),
                                     1000 * pow(x[1], 2)};
        },
        3, 1);
}

/// Whether the point j of the columns (n, N) lies in the box of Z up to tol.
bool in_box(const Zonotope &Z, const std::vector<double> &points, int64_t N, int64_t j,
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
void check_times(const ReachSet &R, double t_final, const std::string &b) {
    check(R.time.size() == R.time_point.size() && R.time_point.size() == R.time_interval.size() + 1,
          b + ": a time and a point set per step boundary");
    check(R.time.front() == 0.0 && R.time.back() >= t_final, b + ": from 0 to t_final or beyond");
    bool increasing = true;
    for (std::size_t k = 1; k < R.time.size(); ++k)
        increasing = increasing && R.time[k] > R.time[k - 1];
    check(increasing, b + ": the times increase");
}

/// Trajectories of a system with constant inputs stay in the sets over the step they are in, and in
/// the sets at the step times.
void simulations_stay_in_the_enclosures(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const Zonotope X0(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    const Zonotope U(Tensor({0.0}), Tensor({{0.1}}));
    const double T = 1.0, h = 1e-3;
    const ReachSet R =
        sys.reach({.initial_set = X0, .input_set = U, .t_final = T}, alg::LinearizationAdaptive{});
    check_times(R, T, b);
    // trajectories from random points of the start set, with constant random inputs
    cora::Rng rng(3);
    const std::vector<Tensor> x =
        sys.simulate_random({.initial_set = Set(X0), .input_set = Set(U), .t_final = T},
                            alg::RungeKutta4{.time_step = h}, 20, rng);
    // each sample lies in the set over the step that contains its time
    int outside = 0;
    std::size_t k = 0;
    for (std::size_t s = 0; s < x.size(); ++s) {
        const double t = double(s) * h;
        while (k + 1 < R.time_interval.size() && R.time[k + 1] < t) ++k;
        const std::vector<double> d = x[s].data();
        for (int j = 0; j < 20; ++j) outside += !in_box(R.time_interval[k], d, 20, j, 1e-9);
    }
    check(outside == 0,
          b + ": " + std::to_string(outside) + " points left the sets over the steps");
}

/// A damped oscillator with a weak nonlinearity: the sets contain the trajectories of the center
/// and of the corners of the initial set at the end, and a linear system is refused.
void a_weak_nonlinearity_and_a_linear_system(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{-x[0] + x[1], -x[0] - x[1] + 0.05 * x[1] * x[1]};
        },
        2);
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const double T = 2.0;
    const ReachSet R = sys.reach({.initial_set = X0, .t_final = T}, alg::LinearizationAdaptive{});
    check_times(R, T, b);
    // the corners of the start set, simulated to the end
    cora::Rng rng(5);
    const Tensor x0 = X0.rand_point(40, rng, "extreme");
    const std::vector<Tensor> x =
        sys.simulate({.initial_state = x0, .t_final = T}, alg::RungeKutta4{.time_step = 1e-3});
    int outside = 0;
    for (int j = 0; j < 40; ++j)
        outside += !in_box(R.time_interval.back(), x.back().data(), 40, j, 1e-9);
    check(outside == 0, b + ": " + std::to_string(outside) + " end points outside");
    // a linear system has no error to tune to: as CORALean's, the loop runs out of fuel
    const NonlinearSys linear(
        [](const std::vector<Expr> &x) { return std::vector<Expr>{-x[0] + x[1], -x[0] - x[1]}; },
        2);
    check(test::raises(
              [&] {
                  linear.reach({.initial_set = X0, .t_final = 1.0},
                               alg::LinearizationAdaptive{.fuel = 50});
              },
              Error::not_converged),
          b + ": a linear system is refused");
}

/// The Robertson system: the first steps enclose a trajectory, and the width of the sum of the
/// concentrations over [0, 1] is tiny (the sum is conserved).
void robertson_sums_stay_conserved(const std::string &b) {
    const NonlinearSys sys = robertson();
    const Zonotope X0(Tensor({1.0, 0.0, 0.0}), Tensor::zeros({3, 0}));
    const Zonotope U(Tensor({0.0}), Tensor::zeros({1, 0}));
    const ReachSet R = sys.reach({.initial_set = X0, .input_set = U, .t_final = 1.0},
                                 alg::LinearizationAdaptive{.zeta_z = 0.001});
    check_times(R, 1.0, b);
    const Interval last = R.time_interval.back().interval();
    double lo = 0, hi = 0;
    for (const double v : last.inf().data()) lo += v;
    for (const double v : last.sup().data()) hi += v;
    check(lo < 1 + 1e-9 && hi > 1 - 1e-9, b + ": the sum 1 is inside");
    check(hi - lo < 1e-3, b + ": the width of the sum is " + std::to_string(hi - lo));
    // the trajectory of (1, 0, 0), sampled every 1e-3
    const std::vector<Tensor> x = sys.simulate(
        {.initial_state = Tensor({{1.0}, {0.0}, {0.0}}), .input = Tensor({{0.0}}), .t_final = 1.0},
        alg::RungeKutta4{.time_step = 1e-5});
    int outside = 0;
    std::size_t k = 0;
    for (std::size_t s = 0; s < x.size(); s += 100) {
        const double t = double(s) * 1e-5;
        while (k + 1 < R.time_interval.size() && R.time[k + 1] < t) ++k;
        outside += !in_box(R.time_interval[k], x[s].data(), 1, 0, 1e-8);
    }
    check(outside == 0,
          b + ": the trajectory leaves the sets " + std::to_string(outside) + " times");
}

/// The sums go from the last entry to the first, the gains and the step size follow their
/// equations: the gain that explains an error ratio is recovered, the step size is the horizon
/// times a power of the factor.
void tuning_formulas_hold(const std::string &b) {
    check(priv_sum_up({1e16, 1.0, 1.0}) == 1e16 + 2.0, b + ": the sum starts at the end");
    check(priv_pow(0.5, 3) == 0.125 && priv_pow(2.0, 0) == 1.0, b + ": powers by multiplication");
    check(priv_norm_up({3.0, -4.0}) >= 5.0 && priv_norm_up({3.0, -4.0}) < 5.0 + 1e-12,
          b + ": the norm of the magnitudes");
    // the ratio of the errors at the horizon and at deltat for the gain 0.3 gives 0.3 back
    const alg::LinearizationAdaptive alg;
    const double decr = alg.decr_factor, lim = priv_pow(decr, 1), deltat = 0.01, horizon = 0.1;
    const double ratio = priv_varphi_total(0.3, lim, deltat, horizon, decr);
    const double gain = priv_varphi_est(horizon, deltat, {1.0}, {ratio}, decr, {1}, 0, alg.fuel);
    check(std::abs(gain - 0.3) < 1e-3, b + ": the gain is recovered, " + std::to_string(gain));
    // the step size is deltat times a power of the factor, not above deltat
    const double dt = priv_optimal_delta_t(decr, alg.zeta_z, 0.5, 1e-3, 0.1, 0.5, 1.0);
    const double steps = std::log(dt / 0.1) / std::log(decr);
    check(dt <= 0.1 && std::abs(steps - std::round(steps)) < 1e-9,
          b + ": the step size is a power of the factor times deltat");
}

/// The adaptive algorithm has no fixed steps; wrong arguments are described.
void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const Zonotope X0(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    // the fixed-step entry points refuse the algorithm, the constants must lie in (0, 1)
    check(test::raises([&] { sys.init_reach(X0, std::nullopt, alg::LinearizationAdaptive{}); },
                       Error::not_supported),
          b + ": init_reach has fixed steps");
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .t_final = 1.0},
                            alg::LinearizationAdaptive{.zeta_z = 0.0});
              },
              Error::wrong_value),
          b + ": zeta_z 0");
    // gains: exactly three
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .t_final = 1.0},
                            alg::LinearizationAdaptive{.zeta_phi = {0.5, 0.5}});
              },
              Error::wrong_value),
          b + ": two gains");
    check(test::throws([&] {
              sys.reach({.initial_set = X0, .t_final = -1.0}, alg::LinearizationAdaptive{});
          }),
          b + ": a negative horizon");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        simulations_stay_in_the_enclosures(b);
        a_weak_nonlinearity_and_a_linear_system(b);
        robertson_sums_stay_conserved(b);
        tuning_formulas_hold(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinear_sys reach adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
