// simulate - runs of a neural network controlled system, as CORA's neurNetContrSys.simulate
//
// Per control period the network is evaluated at the state of the period's start, and the plant
// runs with that output held constant (and the external input of the run); the solver is the
// plant's fixed-step Runge-Kutta, sampled every time_step, the last step of a period shortened.
//
// Syntax:   x = sys.simulate(x0, u, t_final, time_step);
// Inputs:   x0 - the start points, columns (n, N), u - the external input per run, (n_ext, N) or
//           (n_ext, 1) (none without external inputs), t_final - the horizon, time_step - sampling
// Outputs:  x - x[k] holds the N states at time[k]
// See also: simulate_random, reach
//
// CORA: contDynamics/@neurNetContrSys/simulate.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The times t0, t0 + h, ..., t1 with a shorter last step.
std::vector<double> aux_grid(double t0, double t1, double h) {
    const int steps = priv_num_steps(t1 - t0, h);
    std::vector<double> t;
    t.reserve(std::size_t(steps) + 1);
    for (int i = 0; i < steps; ++i) t.push_back(t0 + i * h);
    t.push_back(t1);
    return t;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ClosedLoopTrajectories NeurNetContrSys::simulate(const Tensor &x0, const std::optional<Tensor> &u,
                                                 double t_final, double time_step) const {
    detail::Entry entry;
    // the start points, the horizon and the external input
    const int64_t n_ext = num_external_inputs();
    CORA_CHECK(x0.defined() && x0.shape().size() == 2 && x0.shape()[0] == dim(),
               Error::dimension_mismatch, [&] {
                   return "simulate: the start points must be columns (" + std::to_string(dim()) +
                          ", N)";
               });
    CORA_CHECK(t_final > 0 && t_final < 1e300 && time_step > 0, Error::wrong_value, [&] {
        return "simulate: t_final and time_step must be positive, got " + std::to_string(t_final) +
               " and " + std::to_string(time_step);
    });
    CORA_CHECK(bool(u) == (n_ext > 0) &&
                   (!u || (u->shape()[0] == n_ext &&
                           (u->shape()[1] == 1 || u->shape()[1] == x0.shape()[1]))),
               Error::dimension_mismatch, [&] {
                   return "simulate: the external input must be columns (" + std::to_string(n_ext) +
                          ", N) or one column, exactly when the system " + "has external inputs";
               });
    // the runs: the output of a period is held, and the period is cut to the sampling
    const int64_t N = x0.shape()[1];
    ClosedLoopTrajectories out;
    out.x = {x0};
    out.time = {0.0};
    Tensor x = x0;
    const std::vector<double> periods = aux_grid(0.0, t_final, dt_);
    for (std::size_t p = 0; p + 1 < periods.size(); ++p) {
        // the network output of the start of the period, held, and the external input
        Tensor input = nn_.evaluate(x);
        if (u) input = Tensor::concat({input, u->broadcast_to({n_ext, N})}, 0);
        const std::vector<double> inner = aux_grid(periods[p], periods[p + 1], time_step);
        for (std::size_t i = 0; i + 1 < inner.size(); ++i) {
            const double h = inner[i + 1] - inner[i];
            x = plant_
                    .simulate({.initial_state = x, .input = input, .t_final = h},
                              alg::RungeKutta4{.time_step = h})
                    .back();
            out.x.push_back(x);
            out.time.push_back(inner[i + 1]);
        }
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
