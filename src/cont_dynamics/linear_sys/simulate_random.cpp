// simulate_random - trajectories from random points of a set, as CORA's linearSys.simulateRandom
//
// Syntax:   x = sys.simulate_random(problem, alg, points, rng);
// Inputs:   problem - initial_set (any set), input_set (any set of constant inputs, needs B; none:
//                     no input), t_final
//           alg - alg::Exact (time_step);  points - the number N of trajectories
//           rng - random numbers (a seed makes them repeatable)
// Outputs:  x - x[k] holds the N points at time k*time_step, as simulate
// See also: simulate, ContSet::rand_point
//
// CORA: contDynamics/@contDynamics/simulateRandom.m

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> LinearSys::simulate_random(const RandomSimulationProblem &problem,
                                               const alg::LinearSimulate &alg, int64_t points,
                                               Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("simulate_random: the problem needs an initial_set"); });
    CORA_CHECK(points >= 1, Error::wrong_value, [&] {
        return "simulate_random: needs at least one point, got " + std::to_string(points);
    });
    const Tensor x0 = (*problem.initial_set)->rand_point(points, rng);
    std::optional<Tensor> u;
    if (problem.input_set) u = (*problem.input_set)->rand_point(points, rng);
    return simulate({.initial_state = x0, .input = u, .t_final = problem.t_final}, alg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
