// test_linear_sys_post - post: the steps after init_reach of standard and wrapping_free
//
// Every step encloses the trajectories, and an algorithm without such steps is refused.
//
// CORA: contDynamics/@linearSys/post.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const alg::Standard kStandard{.time_step = kStep, .taylor_terms = kTerms};
const alg::WrappingFree kWrappingFree{.time_step = kStep, .taylor_terms = kTerms};

/// The sets of init_reach and of every post enclose the trajectories, with and without inputs, and
/// the wrapping-free sets keep up with the standard ones.
void the_steps_enclose_the_trajectories(const std::string &b) {
    const Tensor A = diagonal(), B = Tensor::eye(2), u = test::column({0.5, -0.5});
    const LinearSys sys(A, B);
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{0.1}, {0.05}}));
    const Tensor w = u + Tensor({{0.1}, {0.05}});
    for (const alg::LinearReach &a :
         {alg::LinearReach(kStandard), alg::LinearReach(kWrappingFree)}) {
        ReachStep s = sys.init_reach(initial_set(), U, u, a);
        for (int k = 1; k <= 3; ++k) {
            // the end of the step k from the corner (2, 2.5) under the constant input w
            const Tensor x = flow(A, k * kStep, test::column({2.0, 2.5})) +
                             A.inv().matmul((A * (k * kStep)).expm() - Tensor::eye(2)).matmul(w);
            check(s.time_point.contains(x), b + ": time point " + std::to_string(k));
            check(s.time_interval.contains(x), b + ": time interval " + std::to_string(k));
            s = sys.post(s.state, a);
        }
    }
}

/// An algorithm without a post, and a different time step than the state came from, are refused.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal(), Tensor::eye(2));
    const ReachStep first = sys.init_reach(initial_set(), origin(2), std::nullopt, kStandard);
    check(test::raises([&] { sys.post(first.state, alg::Decomposition{}); }, Error::not_supported),
          b + ": decomposition has no post");
    check(test::raises([&] { sys.post(first.state, alg::FromStart{.time_step = kStep}); },
                       Error::not_supported),
          b + ": from_start has no post");
    check(test::raises([&] { sys.post(first.state, alg::Standard{.time_step = -1.0}); },
                       Error::wrong_value),
          b + ": a negative step");
    const Tensor on = sys.post(first.state, kStandard).time_interval.center();
    ChecksScope off(false);
    check(test::close(sys.post(first.state, kStandard).time_interval.center(), on, 0),
          b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_steps_enclose_the_trajectories(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys post");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
