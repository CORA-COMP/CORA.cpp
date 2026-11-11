// test_nonlinear_sys_reach_poly - nonlinearSys reach with alg "poly": exact on a linear system,
//
// and simulations of the van der Pol oscillator, forced by an input too, stay inside the enclosures
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA: contDynamics/@contDynamics/private/priv_abstrerr_poly.m
// CORA: contDynamics/@contDynamics/private/priv_precompStatError.m
// PROPERTY: contDynamics/@contDynamics/private/priv_abstrerr_poly.m
// PROPERTY: contDynamics/@contDynamics/private/priv_precompStatError.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
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
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

PolyZonotope start() {
    return PolyZonotope(Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}})));
}

alg::Polynomialization options(double dt) {
    return {.time_step = dt,
            .taylor_terms = 4,
            .zonotope_order = 20,
            .error_order = 5,
            .intermediate_order = 10};
}

/// How often a trajectory point lies outside the box of the set over its step.
int outside(const PolyReachSet &R, const std::vector<Tensor> &x, double dt, double sim_dt) {
    int count = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const auto step =
            std::min(std::size_t(double(k) * sim_dt / dt), R.time_interval.size() - 1);
        const Interval box = R.time_interval[step].interval();
        const std::vector<double> lo = box.inf().data(), hi = box.sup().data(), p = x[k].data();
        const std::size_t N = p.size() / lo.size();
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < lo.size(); ++i)
                count += !(p[i * N + j] >= lo[i] - 1e-9 && p[i * N + j] <= hi[i] + 1e-9) ? 1 : 0;
    }
    return count;
}

/// A linear system x' = A x: the polynomial result agrees with the linear one up to the series.
void a_linear_system_stays_close(const std::string &b) {
    const NonlinearSys sys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-0.1 * x[0] + x[1], -x[0] - 0.1 * x[1]};
        },
        2, 1);
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const PolyReachSet R = sys.reach(
        {.initial_set = PolyZonotope(X0), .t_final = 1.0},
        alg::PolyNonlinearReach{alg::Polynomialization{
            .time_step = 0.05, .taylor_terms = 8, .error_order = 5, .intermediate_order = 10}});
    const ReachSet lin = LinearSys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}))
                             .reach({.initial_set = X0, .t_final = 1.0},
                                    alg::Standard{.time_step = 0.05, .taylor_terms = 8});
    check(R.time_interval.size() == 20 && R.time_point.size() == 21, b + ": 20 steps");
    const std::vector<double> a = R.time_point.back().interval().sup().data();
    const std::vector<double> c = lin.time_point.back().interval().sup().data();
    check(std::abs(a[0] - c[0]) < 1e-6 && std::abs(a[1] - c[1]) < 1e-6,
          b + ": the last set is that of linear_sys");
}

/// Trajectories from random points of the start set stay in the enclosures, also with an input.
void simulations_stay_in_the_enclosures(const std::string &b) {
    const double dt = 0.01, sim_dt = 0.002;
    const NonlinearSys sys = van_der_pol();
    const Zonotope U(Tensor({0.0}), Tensor({{0.1}}));
    const PolyReachSet R = sys.reach({.initial_set = start(), .input_set = U, .t_final = 1.5},
                                     alg::PolyNonlinearReach{options(dt)});
    cora::Rng rng(1);
    const Tensor x0 = start().rand_point(25, rng);
    for (const double u : {-0.1, 0.0, 0.1}) {
        const std::vector<Tensor> x =
            sys.simulate({.initial_state = x0, .input = Tensor({{u}}), .t_final = 1.5},
                         alg::RungeKutta4{.time_step = sim_dt});
        check(outside(R, x, dt, sim_dt) == 0,
              b + ": points left the enclosure, input " + std::to_string(u));
    }
}

/// The step functions: post continues init_reach, and the order limits the generators.
void the_steps_continue_and_reduce(const std::string &b) {
    const NonlinearSys sys = van_der_pol();
    const alg::Polynomialization a = options(0.01);
    const PolyNonlinearStep first = sys.init_reach(start(), std::nullopt, a);
    const PolyNonlinearStep second = sys.post(first, std::nullopt, a);
    check(first.error.shape() == std::vector<int64_t>({2, 1}), b + ": the error is a column");
    check(second.time_point.num_dependent() +
                  second.time_point.independent_generators().shape()[1] <=
              20 * 2,
          b + ": at most order * n generators after post");
    check(second.time_interval.interval().inf().data()[0] < 1.4 + 0.2,
          b + ": a sensible enclosure");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_linear_system_stays_close(b);
        simulations_stay_in_the_enclosures(b);
        the_steps_continue_and_reduce(b);
    });
    // the checks: a wrong dimension and tensor order 2 are refused
    const NonlinearSys sys = van_der_pol();
    check(test::raises(
              [&] {
                  sys.init_reach(PolyZonotope(Zonotope(Tensor({1.0}), Tensor({{0.1}}))),
                                 std::nullopt, options(0.01));
              },
              Error::dimension_mismatch),
          "init_reach: the dimension");
    // tensor order 2 has no static error and is refused
    check(test::raises(
              [&] {
                  sys.init_reach(start(), std::nullopt,
                                 alg::Polynomialization{.time_step = 0.01, .tensor_order = 2});
              },
              Error::not_supported),
          "init_reach: tensor order 2");
    check(test::raises([&] { sys.reach({.t_final = 1.0}, alg::PolyNonlinearReach{options(0.01)}); },
                       Error::wrong_value),
          "reach: no initial set");
    ChecksScope off(false);
    check(sys.init_reach(start(), std::nullopt, options(0.01)).error.shape()[0] == 2,
          "init_reach: valid with the checks off");
    return test::finish("nonlinear_sys reach poly");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
