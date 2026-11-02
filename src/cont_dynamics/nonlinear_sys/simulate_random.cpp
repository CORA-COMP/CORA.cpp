// simulate_random - simulations from random start points, as CORA's simulateRandom
//
// Syntax:   x = sys.simulate_random(problem, alg, points, rng);
// Inputs:   problem - initial_set (any set), t_final;  alg - alg::RungeKutta4
//           points - number of trajectories;  rng - random numbers (seedable)
// Outputs:  x - as simulate, from `points` random points of the initial set
// See also: simulate, ContSet::rand_point
//
// CORA: contDynamics/@contDynamics/simulateRandom.m

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> NonlinearSys::simulate_random(const RandomSimulationProblem &problem,
                                                  const alg::NonlinearSimulate &alg, int64_t points,
                                                  Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("simulate_random: the problem needs an initial_set"); });
    CORA_CHECK(points >= 1, Error::wrong_value, [&] {
        return "simulate_random: needs at least one point, got " + std::to_string(points);
    });
    CORA_CHECK(!problem.input_set.has_value(), Error::not_supported, [] {
        return std::string("simulate_random: a NonlinearSys has no inputs; leave input_set empty");
    });
    const Tensor x0 = (*problem.initial_set)->rand_point(points, rng);
    return simulate({.initial_state = x0, .t_final = problem.t_final}, alg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
