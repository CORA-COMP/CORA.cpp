// test_nonlinear_sys_reach - nonlinearSys reach: exact on a linear system, and simulations of the
//
// van der Pol oscillator stay inside the enclosures
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);
}

Zonotope van_der_pol_start() {
    return Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
}

/// Whether the column j of the points (n, N) lies in the box of Z, up to `tol`.
bool in_box(const Zonotope &Z, const std::vector<double> &points, int64_t N, int64_t j,
            double tol) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double v = points[i * N + j];
        if (!(v >= lo[i] - tol && v <= hi[i] + tol)) return false; // a NaN is outside
    }
    return true;
}

/// x' = A x is what linear_sys solves: the time-point sets agree up to the Taylor series.
void a_linear_system_gives_the_linear_result(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{-0.1 * x[0] + x[1], -x[0] - 0.1 * x[1]};
        },
        2);
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const ReachSet R = sys.reach({.initial_set = X0, .t_final = 1.0},
                                 alg::Linearization{.time_step = 0.05, .taylor_terms = 8});
    const ReachSet linear = LinearSys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}))
                                .reach({.initial_set = X0, .t_final = 1.0},
                                       alg::Standard{.time_step = 0.05, .taylor_terms = 8});
    check(R.time_interval.size() == 20 && R.time_point.size() == 21, b + ": 20 steps");
    const std::vector<double> a = R.time_point.back().interval().sup().data();
    const std::vector<double> c = linear.time_point.back().interval().sup().data();
    check(std::abs(a[0] - c[0]) < 1e-6 && std::abs(a[1] - c[1]) < 1e-6,
          b + ": the last set is that of linear_sys");
}

/// Trajectories from random and from extreme points of the start set stay inside: at the time
/// points in the point sets, and at five times per step in the sets over the step.
void simulations_stay_in_the_enclosures(const std::string &b) {
    const double dt = 0.005;
    const NonlinearSys sys = van_der_pol();
    const Zonotope X0 = van_der_pol_start();
    const ReachSet R = sys.reach({.initial_set = X0, .t_final = 2.0},
                                 alg::Linearization{.time_step = dt, .taylor_terms = 4});
    check(R.time_point.size() == R.time_interval.size() + 1, b + ": a point set more than steps");

    cora::Rng rng(1);
    for (const std::string type : {"standard", "extreme"}) {
        const Tensor x0 = X0.rand_point(30, rng, type);
        const std::vector<Tensor> x = sys.simulate({.initial_state = x0, .t_final = 2.0},
                                                   alg::RungeKutta4{.time_step = dt / 5});
        int outside = 0;
        for (std::size_t k = 0; k < x.size(); ++k) {
            const std::size_t step = std::min(k / 5, R.time_interval.size() - 1);
            const std::vector<double> d = x[k].data();
            for (int j = 0; j < 30; ++j) {
                outside += !in_box(R.time_interval[step], d, 30, j, 1e-9);
                if (k % 5 == 0) outside += !in_box(R.time_point[k / 5], d, 30, j, 1e-9);
            }
        }
        check(outside == 0,
              b + ": " + type + " points left the enclosure " + std::to_string(outside) + " times");
    }
}

/// The enclosures are usable: over a whole period of the oscillation the box stays small.
void the_enclosure_stays_tight(const std::string &b) {
    const ReachSet R =
        van_der_pol().reach({.initial_set = van_der_pol_start(), .t_final = 6.74},
                            alg::Linearization{.time_step = 0.005, .taylor_terms = 4});
    double widest = 0;
    for (const Zonotope &Z : R.time_interval) {
        const Interval box = Z.interval();
        const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
        widest = std::max({widest, hi[0] - lo[0], hi[1] - lo[1]});
    }
    check(widest < 2.5, b + ": the widest box is " + std::to_string(widest));
}

void the_zonotope_order_limits_the_generators(const std::string &b) {
    const ReachSet R = van_der_pol().reach(
        {.initial_set = van_der_pol_start(), .t_final = 1.0},
        alg::Linearization{.time_step = 0.02, .taylor_terms = 4, .zonotope_order = 3});
    int64_t most = 0;
    for (const Zonotope &Z : R.time_point) most = std::max(most, Z.generators().shape()[1]);
    check(most <= 3 * 2, b + ": at most order * n = 6 generators, found " + std::to_string(most));
}

void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const Zonotope X0 = van_der_pol_start();
    check(test::throws([&] {
              sys.reach({.initial_set = X0, .t_final = 1.0}, alg::Linearization{.time_step = 0.0});
          }),
          b + ": time step 0");
    check(test::throws([&] {
              sys.reach({.initial_set = X0, .t_final = 10.0}, alg::Linearization{.time_step = 5.0});
          }),
          b + ": a step that is too large");
    const Zonotope wrong_dim(Tensor({1.0, 2.0, 3.0}), Tensor::eye(3));
    check(test::throws([&] {
              sys.reach({.initial_set = wrong_dim, .t_final = 1.0},
                        alg::Linearization{.time_step = 0.01});
          }),
          b + ": a set of another dimension");
    check(test::throws([&] {
              sys.reach(
                  {.initial_set = X0, .t_final = 1.0},
                  alg::Linearization{.time_step = 0.01, .taylor_terms = 4, .zonotope_order = 0});
          }),
          b + ": zonotope order 0");
}

/// An input set is refused (no inputs); with the checks off the sets are the same.
void checks_and_their_switch() {
    const NonlinearSys sys = NonlinearSys(
        [](const std::vector<Expr> &x) {
            {
                return std::vector<Expr>{{x[1], -x[0]}};
            }
        },
        2);
    const Zonotope X0 = Zonotope(Tensor({1.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .input_set = X0, .t_final = 1},
                            alg::Linearization{.time_step = 0.1});
              },
              Error::not_supported),
          "reach: an input set");
    const ReachProblem problem{.initial_set = X0, .t_final = 0.5};
    const Tensor on =
        sys.reach(problem, alg::Linearization{.time_step = 0.1}).time_point.back().center();
    ChecksScope off(false);
    check(test::close(
              sys.reach(problem, alg::Linearization{.time_step = 0.1}).time_point.back().center(),
              on, 0),
          "reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_linear_system_gives_the_linear_result(b);
        simulations_stay_in_the_enclosures(b);
        if (b == "eigen") the_enclosure_stays_tight(b); // the slowest test, on one backend
        the_zonotope_order_limits_the_generators(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("nonlinearSys reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
