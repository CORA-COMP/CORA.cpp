// nonlinear_sys_dt - the class of discrete-time nonlinear systems: construction and evaluation
//
// The step map f and the output equation g are two NonlinearSys (the second with r components), so
// their derivatives are those of the continuous class. Without an output equation, y = x.
//
// See also: cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h, reach.cpp
//
// CORA: contDynamics/@nonlinearSysDT/nonlinearSysDT.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Laura Luetzow
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sampling time of a system: a positive finite number (CORA's inputArgsCheck of dt).
double aux_checked_time_step(double time_step) {
    detail::Entry entry;
    CORA_CHECK(time_step > 0 && time_step < 1e300, Error::wrong_input_in_constructor, [&] {
        return "NonlinearSysDT: the sampling time must be positive and finite, got " +
               std::to_string(time_step);
    });
    return time_step;
}

/// The output equation y = x.
InputDynamics aux_identity() {
    return [](const std::vector<Expr> &x, const std::vector<Expr> &) { return x; };
}

/// f(x) as f(x, u) without inputs.
InputDynamics aux_without_inputs(const Dynamics &f) {
    return [f](const std::vector<Expr> &x, const std::vector<Expr> &) { return f(x); };
}

} // namespace


// ===========================================  MAIN  =========================================== //

NonlinearSysDT::NonlinearSysDT(const Dynamics &f, int64_t n, double time_step)
    : NonlinearSysDT(aux_without_inputs(f), n, 0, time_step) {}

NonlinearSysDT::NonlinearSysDT(const InputDynamics &f, int64_t n, int64_t m, double time_step)
    : NonlinearSysDT(f, n, m, time_step, aux_identity(), n) {}

NonlinearSysDT::NonlinearSysDT(const InputDynamics &f, int64_t n, int64_t m, double time_step,
                               const InputDynamics &g, int64_t r)
    : state_(f, n, m), output_(g, n, m, r), time_step_(aux_checked_time_step(time_step)) {}

// The rows of the points run through f: the state and the output equation are evaluated alike.
Tensor NonlinearSysDT::next_state(const Tensor &x, const std::optional<Tensor> &u) const {
    return state_.dynamics(x, u);
}

Tensor NonlinearSysDT::output(const Tensor &x, const std::optional<Tensor> &u) const {
    return output_.dynamics(x, u);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
