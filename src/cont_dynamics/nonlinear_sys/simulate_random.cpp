// simulate_random - simulations from random start points, as CORA's simulateRandom
//
// Syntax:   x = sys.simulate_random(problem, alg, points, rng);
// Inputs:   problem - initial_set, input_set (a random constant input per trajectory), t_final
//           alg - alg::RungeKutta4
//           points - number of trajectories
//           rng - random numbers (seedable)
// Outputs:  x - as simulate, from `points` random points of the initial set
// See also: simulate, ContSet::rand_point
//
// CORA: contDynamics/@contDynamics/simulateRandom.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

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
    CORA_CHECK(m_ > 0 || !problem.input_set, Error::not_supported, [] {
        return std::string("simulate_random: the system has no inputs; leave input_set empty");
    });
    CORA_CHECK(m_ == 0 || (problem.input_set && (*problem.input_set)->dim() == m_),
               Error::dimension_mismatch, [&] {
                   return "simulate_random: the input_set must be a set of the " +
                          std::to_string(m_) + " inputs";
               });
    // the start points, then one constant input per trajectory
    const Tensor x0 = (*problem.initial_set)->rand_point(points, rng);
    std::optional<Tensor> u;
    if (m_ > 0) u = (*problem.input_set)->rand_point(points, rng);
    return simulate({.initial_state = x0, .input = u, .t_final = problem.t_final}, alg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
