// test_linear_sys_reach_from_start - the algorithms from_start and decomposition of reach
//
// Exact time points, sets that enclose random simulations (with the curvature of the input in
// every step), and a partition that loses only the coupling between the blocks.
//
// CORA: contDynamics/@linearSys/private/priv_reach_fromStart.m
// CORA: contDynamics/@linearSys/private/priv_reach_decomp.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"
#include "cora/global/rng.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A damped oscillator coupled to a decaying state, and a set whose generators couple them.
Tensor oscillator() { return Tensor({{-0.1, 1.0, 0.0}, {-1.0, -0.1, 0.0}, {0.0, 0.0, -0.5}}); }
const std::vector<std::array<int64_t, 2>> kBlocks{{0, 1}, {2, 2}};

Zonotope separate_set() {
    return {test::column({1.0, 0.5, -1.0}),
            Tensor({{0.2, 0.0, 0.0}, {0.0, 0.1, 0.0}, {0.0, 0.0, 0.3}})};
}

Zonotope coupled_set() {
    return {test::column({1.0, 0.5, -1.0}),
            Tensor({{0.2, 0.0, 0.1}, {0.0, 0.1, 0.0}, {0.0, 0.0, 0.3}})};
}

/// The boxes of two sets agree up to tol.
bool same_box(const Zonotope &a, const Zonotope &b, double tol) {
    return test::close(a.interval().inf(), b.interval().inf(), tol) &&
           test::close(a.interval().sup(), b.interval().sup(), tol);
}

/// Without an input the time points are e^{A k dt} X0, as for every algorithm.
void the_time_points_are_exact(const std::string &b) {
    const LinearSys sys(oscillator());
    const ReachProblem problem{.initial_set = separate_set(), .t_final = 1.0};
    const ReachSet R = sys.reach(problem, alg::FromStart{.time_step = 0.1, .taylor_terms = 8});
    check(R.time_point.size() == 11 && R.time_interval.size() == 10, b + ": the number of sets");
    for (std::size_t k = 0; k < R.time_point.size(); ++k) {
        const Tensor flow_k = (oscillator() * (0.1 * double(k))).expm();
        check(same_box(R.time_point[k], separate_set().linear_map(flow_k), 1e-12),
              b + ": time point " + std::to_string(k));
    }
    // the time points do not depend on how the state is split, without coupling
    const ReachSet D = sys.reach(
        problem, alg::Decomposition{.time_step = 0.1, .taylor_terms = 8, .partition = kBlocks});
    for (std::size_t k = 0; k < R.time_point.size(); ++k) {
        check(same_box(D.time_point[k], R.time_point[k], 1e-12),
              b + ": decomposed point " + std::to_string(k));
        if (k < R.time_interval.size())
            check(same_box(D.time_interval[k], R.time_interval[k], 1e-12),
                  b + ": decomposed interval " + std::to_string(k));
    }
}

/// A coupling between the blocks is lost by the partition: the sets grow, and still contain the
/// old.
void the_partition_loses_the_coupling(const std::string &b) {
    const LinearSys sys(oscillator());
    const ReachProblem problem{.initial_set = coupled_set(), .t_final = 1.0};
    const ReachSet R = sys.reach(problem, alg::FromStart{.time_step = 0.1, .taylor_terms = 8});
    const ReachSet D = sys.reach(
        problem, alg::Decomposition{.time_step = 0.1, .taylor_terms = 8, .partition = kBlocks});
    // the product of the blocks holds the coupled set: a larger support in every direction
    const std::vector<std::vector<double>> directions{{1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {-1, 2, 1}};
    for (std::size_t k = 0; k < R.time_interval.size(); ++k)
        for (const std::vector<double> &d : directions)
            check(support(D.time_interval[k], d) >= support(R.time_interval[k], d) - 1e-12,
                  b + ": the interval grows");
    // along (1, 0, -1) the coupling of the first and the last state cancels in the coupled set
    const std::vector<double> across{1, 0, -1};
    const bool larger =
        support(D.time_interval[0], across) > support(R.time_interval[0], across) + 1e-3;
    check(larger, b + ": the coupling matters");
}

/// Random trajectories, at the time points and in between, stay in the sets; the oscillator and the
/// large constant input make the curvature of the input matter at every step.
void simulations_stay_inside(const std::string &b) {
    const Tensor A = oscillator(), B = Tensor::eye(3);
    const LinearSys sys(A, B);
    const Zonotope X0 = coupled_set();
    const Zonotope U(test::column({1.0, 2.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.2}, {0.1, 0.1}}));
    const double dt = 0.5;
    const ReachProblem problem{.initial_set = X0, .input_set = U, .t_final = 3.0};
    for (const alg::LinearReach &a :
         {alg::LinearReach(alg::FromStart{.time_step = dt, .taylor_terms = 8}),
          alg::LinearReach(
              alg::Decomposition{.time_step = dt, .taylor_terms = 8, .partition = kBlocks})}) {
        const ReachSet R = sys.reach(problem, a);
        cora::Rng rng(11);
        // samples every dt / 2: the even ones are time points, the odd ones lie inside a step
        const std::vector<Tensor> x =
            sys.simulate_random({.initial_set = Set(X0), .input_set = Set(U), .t_final = 3.0},
                                alg::Exact{.time_step = dt / 2}, 40, rng);
        for (std::size_t k = 0; k < x.size(); ++k) {
            if (k % 2 == 0)
                check(R.time_point[k / 2].contains(x[k]),
                      b + ": time point " + std::to_string(k / 2));
            else
                check(R.time_interval[k / 2].contains(x[k]),
                      b + ": inside step " + std::to_string(k / 2));
        }
    }
}

/// A constant input (U a single point) accumulates over the steps: x' = -x + 1 tends to 1, where
/// CORA's fromStart adds it once (docs/cora-differences.md).
void a_constant_input_accumulates(const std::string &b) {
    const LinearSys sys(Tensor({{-1.0}}), Tensor({{1.0}}));
    const ReachProblem problem{.initial_set = origin(1), .input_set = point({1.0}), .t_final = 5.0};
    for (const alg::LinearReach &a :
         {alg::LinearReach(alg::FromStart{.time_step = 0.5, .taylor_terms = 8}),
          alg::LinearReach(
              alg::Decomposition{.time_step = 0.5, .taylor_terms = 8, .partition = {{0, 0}}})}) {
        const ReachSet R = sys.reach(problem, a);
        check(R.time_point.back().contains(test::column({1.0 - std::exp(-5.0)})),
              b + ": the state at 5 is 1 - e^-5");
    }
}

/// A partition that is empty, has a gap or stops short is refused; with the checks off, the same
/// sets.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(oscillator());
    const ReachProblem problem{.initial_set = separate_set(), .t_final = 0.5};
    const auto run = [&](const std::vector<std::array<int64_t, 2>> &partition) {
        sys.reach(problem, alg::Decomposition{.time_step = 0.1, .partition = partition});
    };
    check(test::raises([&] { run({}); }, Error::wrong_value), b + ": no partition");
    check(test::raises([&] { run({{0, 0}, {2, 2}}); }, Error::wrong_value), b + ": a gap");
    check(test::raises([&] { run({{0, 1}}); }, Error::wrong_value), b + ": it stops short");
    check(test::raises([&] { sys.reach(problem, alg::FromStart{.time_step = 0.0}); },
                       Error::wrong_value),
          b + ": time_step 0");
    const Tensor on =
        sys.reach(problem, alg::FromStart{.time_step = 0.1}).time_interval.back().center();
    ChecksScope off(false);
    check(test::close(
              sys.reach(problem, alg::FromStart{.time_step = 0.1}).time_interval.back().center(),
              on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_time_points_are_exact(b);
        the_partition_loses_the_coupling(b);
        simulations_stay_inside(b);
        a_constant_input_accumulates(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys reach from_start");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
