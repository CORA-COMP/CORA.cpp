// benchmark_neur_net_double_pendulum - the double pendulum plant of the two programs
//
// The model of models/Cora/doublePendulum.m, which both scripts use.
//
// Syntax:   const NonlinearSys plant = cora::bench::double_pendulum_plant();
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_05_doublePendulum_lessRobust.m
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

/// models/Cora/doublePendulum.m
inline NonlinearSys double_pendulum_plant() {
    constexpr double m = 0.5, L = 0.5, c = 0, g = 1;
    return {[=](const std::vector<Expr> &x, const std::vector<Expr> &u) {
                const Expr d = x[0] - x[1], s = cos(d);
                // the force terms of the first link, shared by both accelerations
                const Expr B = g / L * sin(x[0]) - 0.5 * pow(x[3], 2) * sin(d) +
                               (u[0] - c * x[2]) / (2 * L * L * m);
                const Expr T = pow(x[2], 2) * sin(d) - s * B + g / L * sin(x[1]) +
                               (u[1] - c * x[3]) / (L * L * m);
                const Expr den = 0.5 * pow(cos(d), 2) - 1.0;
                return std::vector<Expr>{x[2], x[3], s * T / (2.0 * den) + B, -1.0 * T / den};
            },
            4, 2};
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
