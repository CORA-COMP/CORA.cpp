// test_nonlin_das_sys_reach - nonlinDASys reach: exact when linear, enclosing when nonlinear
//
// A linear differential-algebraic system gives the sets of its reduced linear system. For
// x' = y, 0 = y - x^2 (CORALean's counterexample to the sign of the algebraic error) and a coupled
// system of two parts whose second part drives the first through g, the sets enclose simulated
// trajectories, also when the Lagrange remainder is evaluated per subsystem on a decomposition.
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinDASys/initReach.m
// CORA: contDynamics/@nonlinDASys/linReach.m
// CORA: contDynamics/@nonlinDASys/post.m
// PROPERTY: contDynamics/@nonlinDASys/initReach.m
// PROPERTY: contDynamics/@nonlinDASys/linReach.m
// PROPERTY: contDynamics/@nonlinDASys/post.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vars = std::vector<Expr>;

/// x' = y, 0 = y - x^2.
NonlinDASys square_system() {
    return NonlinDASys(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0] * x[0]}; }, 1, 0,
        1);
}

/// x' = -x + y, 0 = y - x/2 - u: x' = -x/2 + u.
NonlinDASys linear_system() {
    return NonlinDASys(
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{-1 * x[0] + y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &u) { return Vars{y[0] - 0.5 * x[0] - u[0]}; },
        1, 1, 1);
}

/// x1' = y1, x2' = y2, 0 = y1 - y2, 0 = y2 - x2^2; one unused input. y1 = y2 = x2^2.
NonlinDASys coupled_system() {
    return NonlinDASys([](const Vars &, const Vars &y, const Vars &) { return Vars{y[0], y[1]}; },
                       [](const Vars &x, const Vars &y, const Vars &) {
                           return Vars{y[0] - y[1], y[1] - x[1] * x[1]};
                       },
                       2, 1, 2);
}

/// Its subsystems: (x1, y1) reading y2 as its input, and (x2, y2) with an unused input.
DaeDecomposition coupled_decomposition() {
    const NonlinDASys first(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &, const Vars &y, const Vars &u) { return Vars{y[0] - u[0]}; }, 1, 1, 1);
    const NonlinDASys second(
        [](const Vars &, const Vars &y, const Vars &) { return Vars{y[0]}; },
        [](const Vars &x, const Vars &y, const Vars &) { return Vars{y[0] - x[0] * x[0]}; }, 1, 1,
        1);
    const Tensor zero = Tensor({{0.0}});
    return {
        {{first, Tensor({{1.0, 0.0}}), Tensor({{1.0, 0.0}}), zero, Tensor({{0.0, 1.0}}), zero},
         {second, Tensor({{0.0, 1.0}}), Tensor({{0.0, 1.0}}), zero, Tensor({{0.0, 0.0}}), zero}},
        Tensor({0.0})};
}

/// Whether the column j of the points (n, N) lies in the box of Z, up to `tol`.
bool in_box(const Zonotope &Z, const Tensor &points, int64_t j, double tol) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    const std::vector<double> p = points.slice(-1, j, j + 1).data();
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (!(p[i] >= lo[i] - tol && p[i] <= hi[i] + tol)) return false; // a NaN is outside
    return true;
}

/// The number of times trajectories of N random points of X0 (standard and extreme) leave the sets
/// of R: at the time points, and at the middle of every step in the sets over the step.
int escapes(const NonlinDASys &sys, const Zonotope &X0, const Tensor &y_guess, const Tensor &u,
            const DaeReachSet &R, double t_final, int N) {
    int outside = 0;
    cora::Rng rng(3);
    for (const std::string type : {"standard", "extreme"}) {
        const Tensor x0 = X0.rand_point(N, rng, type);
        for (int j = 0; j < N; ++j) {
            // sampled at half steps: the even samples are the time points
            const DaeTrajectory T = sys.simulate(x0.slice(-1, j, j + 1), y_guess, u, t_final,
                                                 alg::RungeKutta4{.time_step = R.time_step / 2});
            for (std::size_t k = 0; k < R.time_point.size(); ++k)
                outside += !in_box(R.time_point[k], T.x, int64_t(2 * k), 1e-9);
            for (std::size_t k = 0; k < R.time_interval.size(); ++k)
                outside += !in_box(R.time_interval[k], T.x, int64_t(2 * k + 1), 1e-9);
        }
    }
    return outside;
}

/// A linear system gives the sets of the reduced linear system x' = -x/2 + u, u in [-0.1, 0.1].
void a_linear_system_gives_the_linear_result(const std::string &b) {
    const Zonotope X0(Tensor({1.0}), Tensor({{0.1}})), U(Tensor({0.0}), Tensor({{0.1}}));
    const DaeReachSet R = linear_system().reach(
        {.initial_set = X0, .input_set = U, .y0_guess = Tensor({0.0}), .t_final = 1.0},
        alg::DaeLinearization{.time_step = 0.05, .taylor_terms = 8});
    const ReachSet linear = LinearSys(Tensor({{-0.5}}), Tensor({{1.0}}))
                                .reach({.initial_set = X0, .input_set = U, .t_final = 1.0},
                                       alg::Standard{.time_step = 0.05, .taylor_terms = 8});
    check(R.time_interval.size() == 20 && R.time_point.size() == 21 && R.algebraic.size() == 20,
          b + ": 20 steps");
    const Interval a = R.time_point.back().interval(), c = linear.time_point.back().interval();
    check(test::close(a.inf(), c.inf(), 1e-6) && test::close(a.sup(), c.sup(), 1e-6),
          b + ": the last set is that of the reduced linear system");
}

/// x' = y, 0 = y - x^2: the trajectories x' = x^2 stay in the sets. CORA's sign of the algebraic
/// error misses them (CORALean: linErrorCora_counterexample).
void trajectories_stay_in_the_sets(const std::string &b) {
    const NonlinDASys sys = square_system();
    const Zonotope X0(Tensor({0.4}), Tensor({{0.2}}));
    const DaeReachSet R = sys.reach({.initial_set = X0, .y0_guess = Tensor({0.0}), .t_final = 1.0},
                                    alg::DaeLinearization{.time_step = 0.01});
    check(escapes(sys, X0, Tensor({0.0}), Tensor::zeros({0, 1}), R, 1.0, 20) == 0,
          b + ": x' = x^2 stays in the enclosures");
}

/// The coupled system with and without a decomposition: both enclose, also x1, which only the
/// coupling y1 = y2 = x2^2 moves and which a subsystem's own blocks of C F^-1 would miss.
void a_decomposition_encloses_the_coupling(const std::string &b) {
    const NonlinDASys sys = coupled_system();
    const Zonotope X0(Tensor({0.1, 0.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    const Zonotope U(Tensor({0.0}), Tensor::zeros({1, 0}));
    const alg::DaeLinearization a{.time_step = 0.01, .zonotope_order = 20};
    const Tensor y_guess = Tensor({0.0, 0.0});
    for (const bool composed : {false, true}) {
        DaeReachProblem problem{
            .initial_set = X0, .input_set = U, .y0_guess = y_guess, .t_final = 0.5};
        if (composed) problem.decomposition = coupled_decomposition();
        const DaeReachSet R = sys.reach(problem, a);
        check(escapes(sys, X0, y_guess, Tensor({0.0}), R, 0.5, 20) == 0,
              b + (composed ? ": a decomposition" : ": the whole system") + " encloses");
    }
}

/// init_reach and post are the steps of reach, and the piecewise constant input u_trans moves the
/// sets.
void the_steps_are_those_of_reach(const std::string &b) {
    const NonlinDASys sys = linear_system();
    const Zonotope X0(Tensor({1.0}), Tensor({{0.1}}));
    const alg::DaeLinearization a{.time_step = 0.05, .taylor_terms = 6};
    const DaeReachProblem problem{.initial_set = X0,
                                  .u_trans = Tensor({{0.0, 1.0}}),
                                  .y0_guess = Tensor({0.0}),
                                  .t_final = 0.1};
    const DaeReachSet R = sys.reach(problem, a);
    const DaeStep first = sys.init_reach(problem, a);
    const DaeStep second = sys.post(first, problem, Tensor({1.0}), a);
    check(test::close(first.time_point.center(), R.time_point[1].center(), 1e-12) &&
              test::close(second.time_point.center(), R.time_point[2].center(), 1e-12),
          b + ": the steps of reach");
    // with u = 1 in the second step the center moves up by about 1 * dt: x' = -x/2 + 1
    check(second.time_point.center().item() > first.time_point.center().item(),
          b + ": the constant input of the second step");
}

/// Wrong arguments are described, and the checks can be switched off.
void wrong_arguments_are_described(const std::string &b) {
    // the problem needs its sets, a step size, the order 2 and an input only for a system with one
    const NonlinDASys sys = square_system();
    const Zonotope X0(Tensor({0.4}), Tensor({{0.2}}));
    const Tensor y_guess = Tensor({0.0});
    const alg::DaeLinearization a{.time_step = 0.01};
    // a problem without the guess of y
    check(test::raises([&] { sys.reach({.initial_set = X0, .t_final = 1.0}, a); },
                       Error::wrong_value),
          b + ": no y0_guess");
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .y0_guess = y_guess, .t_final = 1.0},
                            alg::DaeLinearization{.time_step = 0.0});
              },
              Error::wrong_value),
          b + ": time step 0");
    // only the second order is ported
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .y0_guess = y_guess, .t_final = 1.0},
                            alg::DaeLinearization{.time_step = 0.01, .tensor_order = 3});
              },
              Error::not_supported),
          b + ": tensor order 3");
    check(test::raises(
              [&] {
                  sys.reach(
                      {.initial_set = X0, .input_set = X0, .y0_guess = y_guess, .t_final = 1.0}, a);
              },
              Error::dimension_mismatch),
          b + ": an input set for a system without inputs");
    // a bound on the error that the step exceeds: CORA would split the set and aborts
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .y0_guess = y_guess, .t_final = 1.0},
                            alg::DaeLinearization{
                                .time_step = 0.01, .max_error_x = {1e-12}, .max_error_y = {1e-12}});
              },
              Error::not_supported),
          b + ": the error exceeds max_error");
    // a state that no subsystem owns (the second subsystem is dropped)
    DaeDecomposition dec = coupled_decomposition();
    dec.subsystems.pop_back();
    const Zonotope X2(Tensor({0.1, 0.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    check(test::raises(
              [&] {
                  coupled_system().reach({.initial_set = X2,
                                          .y0_guess = Tensor({0.0, 0.0}),
                                          .t_final = 0.1,
                                          .decomposition = dec},
                                         alg::DaeLinearization{.time_step = 0.01});
              },
              Error::wrong_value),
          b + ": a state owned by no subsystem");
    // the same problem with the checks on, then off
    const DaeReachProblem problem{.initial_set = X0, .y0_guess = y_guess, .t_final = 0.1};
    const Tensor on = sys.reach(problem, a).time_point.back().center();
    ChecksScope off(false);
    check(test::close(sys.reach(problem, a).time_point.back().center(), on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_linear_system_gives_the_linear_result(b);
        trajectories_stay_in_the_sets(b);
        a_decomposition_encloses_the_coupling(b);
        the_steps_are_those_of_reach(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("nonlinDASys reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
