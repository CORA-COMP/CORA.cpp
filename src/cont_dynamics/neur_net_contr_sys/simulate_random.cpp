// simulate_random - runs of a neural network controlled system from random points
//
// As CORA's simulateRandom of type standard: a fraction of the start points are corners of the
// initial set, the others random points of it; every run gets one random constant external
// input of the input set. The runs are those of simulate.
//
// Syntax:   x = sys.simulate_random(X0, U, t_final, time_step, points, fraction_vertices, rng);
// Inputs:   X0 - the initial set, U - the external input set (none without external inputs),
//           points - the number of runs, fraction_vertices - the share that starts at corners
// Outputs:  x - as simulate
// See also: simulate, verify
//
// CORA: contDynamics/@contDynamics/simulateRandom.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ClosedLoopTrajectories NeurNetContrSys::simulate_random(const Zonotope &X0,
                                                        const std::optional<Zonotope> &U,
                                                        double t_final, double time_step,
                                                        int64_t points, double fraction_vertices,
                                                        Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(points >= 1 && fraction_vertices >= 0 && fraction_vertices <= 1, Error::wrong_value,
               [&] {
                   return "simulate_random: needs at least one point and a fraction in [0, 1], "
                          "got " +
                          std::to_string(points) + " and " + std::to_string(fraction_vertices);
               });
    // corners first, then inner points
    const int64_t corners = static_cast<int64_t>(std::round(fraction_vertices * double(points)));
    std::vector<Tensor> starts;
    if (corners > 0) starts.push_back(X0.rand_point(corners, rng, "extreme"));
    if (points > corners) starts.push_back(X0.rand_point(points - corners, rng, "standard"));
    std::optional<Tensor> u;
    if (num_external_inputs() > 0) {
        CORA_CHECK(U.has_value(), Error::wrong_value, [] {
            return std::string("simulate_random: the system has external inputs; give their set");
        });
        u = U->rand_point(points, rng, "standard");
    }
    return simulate(Tensor::concat(starts, 1), u, t_final, time_step);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
