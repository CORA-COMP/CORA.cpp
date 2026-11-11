// neur_net_contr_sys_fixture - the system the neurNetContrSys tests share
//
// A van der Pol oscillator whose second input is driven by a tanh network and whose first input is
// external, a start set, an input set, and the options of reach.
//
// Syntax:   const NeurNetContrSys sys(fixture::plant(), fixture::network(), 0.05);
// See also: test_neur_net_contr_sys_reach.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace fixture {

using namespace cora;

/// x' = (x2 + u_ext, (1 - x1^2) x2 - x1 + u_nn): the network drives the first input.
inline NonlinearSys plant() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1] + u[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 2);
}

/// Two tanh layers with 3 hidden neurons and one output.
inline NeuralNetwork network() {
    return NeuralNetwork(
        {{Tensor({{0.8, -0.5}, {0.3, 0.9}, {-0.6, 0.4}}), Tensor({0.1, -0.2, 0.0})},
         {Tensor({{0.7, -0.4, 0.5}}), Tensor({0.05})}},
        {Activation::tanh, Activation::tanh});
}

inline Zonotope X0() { return Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.04}})); }
inline Zonotope U() { return Zonotope(Tensor({0.0}), Tensor({{0.01}})); }

/// The plant's algorithm: 0.025 steps, four Taylor terms, order 20.
inline ClosedLoopOptions options() {
    ClosedLoopOptions o;
    o.plant = {.time_step = 0.025, .taylor_terms = 4, .zonotope_order = 20};
    return o;
}

} // namespace fixture

// ---------------------------------------  END OF CODE  ---------------------------------------- //
