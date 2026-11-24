// test_nonlinear_sys_dt_reach - nonlinearSysDT reach, post and outputSet: hand cases and enclosures
//
// A linear system has no linearization error, so its sets are written out by hand; for a nonlinear
// system with a nonlinear output equation, the sampled trajectories of every tensor order lie in
// the sets. The pinned oracle serves no nonlinearSysDT operation (docs/oracle-requests.md), so
// there is no oracle comparison yet.
//
// CORA: contDynamics/@nonlinearSysDT/reach.m
// CORA: contDynamics/@nonlinearSysDT/post.m
// CORA: contDynamics/@nonlinearSysDT/linReach.m
// CORA: contDynamics/@nonlinearSysDT/nonlinearSysDT.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linearize.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linError_mixed_noInt.m
// CORA: contDynamics/@nonlinearSysDT/private/priv_linError_thirdOrder.m
// CORA: contDynamics/@contDynamics/outputSet.m
// PROPERTY: contDynamics/@nonlinearSysDT/reach.m
// PROPERTY: contDynamics/@nonlinearSysDT/linReach.m
// PROPERTY: contDynamics/@contDynamics/outputSet.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x+ = (x0 + 0.1 x1 + u, x1 - 0.1 sin x0), the output y = x0 x1 + x0.
NonlinearSysDT aux_pendulum() {
    return NonlinearSysDT(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[0] + 0.1 * x[1] + u[0], x[1] - 0.1 * sin(x[0])};
        },
        2, 1, 0.1,
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{x[0] * x[1] + x[0]};
        },
        1);
}

Zonotope aux_start(double radius) {
    return Zonotope(test::vec({1.0, 0.5}), Tensor({{radius, 0.0}, {0.0, radius}}));
}

/// A number in [-1, 1].
double aux_uniform(Rng &rng) {
    double v = 0;
    rng.uniform(&v, 1, -1.0, 1.0);
    return v;
}

/// Whether the point (a vector) lies in the box hull of Z.
bool aux_in_hull(const Zonotope &Z, const Tensor &p) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data(), v = p.data();
    for (std::size_t i = 0; i < v.size(); ++i)
        if (v[i] < lo[i] - 1e-12 || v[i] > hi[i] + 1e-12) return false;
    return true;
}

/// x+ = 0.5 x + u from [0.8, 1.2] with u in [-0.1, 0.1]: [0.3, 0.7] after one step, exactly.
void a_linear_system_has_no_error(const std::string &b) {
    const NonlinearSysDT sys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{0.5 * x[0] + u[0]};
        },
        1, 1, 0.25);
    const Zonotope X0(test::vec({1.0}), Tensor({{0.2}})), U(test::vec({0.0}), Tensor({{0.1}}));
    const ReachSetDT R = sys.reach({.initial_set = X0, .input_set = U, .t_final = 0.5},
                                   alg::DiscreteLinearization{.zonotope_order = 5});
    check(R.states.size() == 3 && R.outputs.size() == 3 && R.time_step == 0.25,
          b + ": three time points");
    const Interval I1 = R.states[1].interval();
    check(test::close(I1.inf(), {0.3}, 1e-12) && test::close(I1.sup(), {0.7}, 1e-12),
          b + ": the first step");
    // the second step: 0.5 [0.3, 0.7] + [-0.1, 0.1] = [0.05, 0.45]
    const Interval I2 = R.states[2].interval();
    check(test::close(I2.inf(), {0.05}, 1e-12) && test::close(I2.sup(), {0.45}, 1e-12),
          b + ": the second step");
    // y = x: the outputs are the states, and there is no error
    check(test::close(R.outputs[1].center(), R.states[1].center(), 0) &&
              test::close(R.error_states[1].interval().sup(), {0.0}, 1e-15),
          b + ": the output set is the state set, the error is 0");
}

/// Sampled trajectories lie in the states and the outputs, for each order of the two equations.
void trajectories_lie_in_the_sets(const std::string &b) {
    const NonlinearSysDT sys = aux_pendulum();
    const Zonotope X0 = aux_start(0.05), U(test::vec({0.0}), Tensor({{0.01}}));
    for (int order : {2, 3})
        for (int order_out : {2, 3}) {
            const ReachSetDT R =
                sys.reach({.initial_set = X0, .input_set = U, .t_final = 0.5},
                          alg::DiscreteLinearization{.tensor_order = order,
                                                     .tensor_order_output = order_out});
            Rng rng(7);
            bool inside = true;
            for (int s = 0; s < 40; ++s) {
                // a start point and a constant input
                Tensor x =
                    test::vec({1.0 + 0.05 * aux_uniform(rng), 0.5 + 0.05 * aux_uniform(rng)});
                const Tensor u = test::vec({0.01 * aux_uniform(rng)});
                for (std::size_t k = 0; k < R.states.size(); ++k) {
                    inside = inside && aux_in_hull(R.states[k], x) &&
                             aux_in_hull(R.outputs[k], sys.output(x, u));
                    x = sys.next_state(x, u);
                }
            }
            check(inside, b + ": trajectories in the sets, orders " + std::to_string(order) + "/" +
                              std::to_string(order_out));
        }
}

/// The sets of post and reach agree, and an output skipped gives the states.
void post_and_the_output_option(const std::string &b) {
    const NonlinearSysDT sys = aux_pendulum();
    const Zonotope X0 = aux_start(0.05), U(test::vec({0.0}), Tensor({{0.01}}));
    // the first set of reach after the start is the step of post
    const alg::DiscreteLinearization alg{};
    const ReachSetDT R = sys.reach({.initial_set = X0, .input_set = U, .t_final = 0.1}, alg);
    const DiscreteStep s = sys.post(X0, U, alg);
    check(test::close(s.set.center(), R.states[1].center(), 0) &&
              test::close(s.set.generators(), R.states[1].generators(), 0),
          b + ": post is the first step of reach");
    const ReachSetDT S = sys.reach({.initial_set = X0, .input_set = U, .t_final = 0.1},
                                   alg::DiscreteLinearization{.compute_output_set = false});
    check(test::close(S.outputs[1].center(), S.states[1].center(), 0),
          b + ": no output set: the states");
    // the output equation has one component and is not linear
    check(sys.output_dim() == 1 && !sys.output_is_linear(), b + ": the output equation");
}

void wrong_arguments_are_described(const std::string &b) {
    const NonlinearSysDT sys = aux_pendulum();
    const Zonotope X0 = aux_start(0.05);
    check(test::raises([&] { sys.reach({.t_final = 0.1}, alg::DiscreteLinearization{}); },
                       Error::wrong_value),
          b + ": no initial set");
    check(test::raises(
              [&] {
                  // a start set of dimension 1 for a system of dimension 2
                  sys.reach(
                      {.initial_set = Zonotope(test::vec({1.0}), Tensor({{0.1}})), .t_final = 0.1},
                      alg::DiscreteLinearization{});
              },
              Error::not_supported),
          b + ": a set of the wrong dimension");
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = X0, .t_final = 0.1},
                            alg::DiscreteLinearization{.tensor_order = 4});
                  // a tensor order other than 2 and 3
              },
              Error::wrong_value),
          b + ": tensor_order 4");
    // a sampling time of 0
    check(test::raises(
              [&] {
                  NonlinearSysDT([](const std::vector<Expr> &x) { return std::vector<Expr>{x[0]}; },
                                 1, 0.0);
              },
              Error::wrong_input_in_constructor),
          b + ": a zero sampling time");
}

void checks_and_their_switch() {
    const NonlinearSysDT sys = aux_pendulum();
    const Zonotope X0 = aux_start(0.05);
    const ReachSetDT on =
        sys.reach({.initial_set = X0, .t_final = 0.2}, alg::DiscreteLinearization{});
    ChecksScope off(false);
    const ReachSetDT no =
        sys.reach({.initial_set = X0, .t_final = 0.2}, alg::DiscreteLinearization{});
    check(test::close(no.states.back().center(), on.states.back().center(), 0),
          "reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_linear_system_has_no_error(b);
        trajectories_lie_in_the_sets(b);
        post_and_the_output_option(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("nonlinearSysDT reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
