// benchmark_neur_net_tora - the TORA plant of the three TORA programs
//
// The translational oscillator with rotational actuator, which the scripts of TORA, TORA
// heterogeneous sigmoid and TORA heterogeneous tanh share; the controllers differ in the output
// scaling.
//
// Syntax:   const NonlinearSys plant = cora::bench::tora_plant(1.0, -10.0);
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_03_TORA.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cora.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

/// example_neuralNet_reach_03_TORA.m (the output is u = f(x) - 10) and the heterogeneous ones
/// (u = 11 f(x)).
inline NonlinearSys tora_plant(double gain, double offset) {
    return {[=](const std::vector<Expr> &x, const std::vector<Expr> &u) {
                return std::vector<Expr>{x[1], -1.0 * x[0] + 0.1 * sin(x[2]), x[3],
                                         gain * u[0] + offset};
            },
            4, 1};
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
