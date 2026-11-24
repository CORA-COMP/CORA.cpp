// nonlinear_sys_dt - the discrete-time nonlinear system x(k+1) = f(x(k), u(k)), as nonlinearSysDT
//
// The dynamics f, and the output equation y = g(x, u) (default y = x), are written on symbolic
// states and inputs (cora/global/expr.h) and held as two NonlinearSys, so the Jacobians, Hessians
// and third derivatives that reach needs are those of the continuous class. A single system and a
// single initial set: no batches. The sampling time is part of the system.
//
// Syntax:     NonlinearSysDT sys(f, n, m, dt);  NonlinearSysDT sys(f, n, m, dt, g, r);
//             ReachSetDT R = sys.reach({.initial_set = X0, .input_set = U, .t_final = 1}, alg);
// Operations: reach, post (reach.cpp), output_set (output_set.cpp), simulate (simulate.cpp); the
//             constructors and the evaluation of f and g (nonlinear_sys_dt.cpp)
// See also:   cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h
//
// CORA: contDynamics/@nonlinearSysDT/nonlinearSysDT.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Laura Luetzow
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The results -------------------------------------------------------------------------------------

/// One step of reach: the set of the states after the step, and the set of the linearization error
/// that was added to it (CORA's Rnext and Verror.L_x).
struct DiscreteStep {
    Zonotope set, error;
};

/// The reachable sets of a discrete-time system: states[k] and outputs[k] hold the sets at time
/// k*time_step; the errors of each step are those of the linearization (CORA's Verror.L_x and
/// Verror.L_y, the first being 0).
struct ReachSetDT {
    std::vector<Zonotope> states, outputs;
    std::vector<Zonotope> error_states, error_outputs;
    double time_step = 0;
};

/// What simulate is asked: trajectories from the rows of initial_state (N, n) under a constant
/// input (m) or (N, m), or under the signal input_signal[k] (N, m) or (m) of step k (CORA's u).
struct DiscreteSimulationProblem {
    Tensor initial_state;                       ///< the start points (N, n), required
    std::optional<Tensor> input = std::nullopt; ///< the constant input
    std::optional<std::vector<Tensor>> input_signal = std::nullopt; ///< the input of each step
    double t_final = 0;       ///< the time horizon, a multiple of dt
    bool with_output = false; ///< also the outputs y(k) (CORA's y)
};

/// The states x[k] and, if asked, the outputs y[k] at the times k*time_step, each (N, n), (N, r).
struct DiscreteTrajectory {
    std::vector<Tensor> x, y;
};

// The System --------------------------------------------------------------------------------------

class NonlinearSysDT {
  public:
    /// The system x(k+1) = f(x(k)) of dimension n with sampling time dt; f is called once, here.
    NonlinearSysDT(const Dynamics &f, int64_t n, double time_step);

    /// The system x(k+1) = f(x(k), u(k)) with n states and m inputs; the output is y = x.
    NonlinearSysDT(const InputDynamics &f, int64_t n, int64_t m, double time_step);

    /// The same with the output equation y = g(x, u) of r components.
    NonlinearSysDT(const InputDynamics &f, int64_t n, int64_t m, double time_step,
                   const InputDynamics &g, int64_t r);

    int64_t dim() const { return state_.dim(); }
    int64_t input_dim() const { return state_.input_dim(); }
    int64_t output_dim() const { return output_.output_dim(); }
    double time_step() const { return time_step_; }
    /// f as a NonlinearSys (its Jacobians and derivatives are those of the step map).
    const NonlinearSys &state_equation() const { return state_; }
    /// g as a NonlinearSys with r components.
    const NonlinearSys &output_equation() const { return output_; }
    /// Whether the output equation is linear (all its second derivatives vanish identically).
    bool output_is_linear() const { return output_.is_affine(); }

    /// f at x and the input u (CORA's mFile): rows (N, n) and (N, m) or (m), or vectors.
    Tensor next_state(const Tensor &x, const std::optional<Tensor> &u = std::nullopt) const;
    /// g at x and u (CORA's out_mFile).
    Tensor output(const Tensor &x, const std::optional<Tensor> &u = std::nullopt) const;

    /// The reachable sets from problem.initial_set under problem.input_set at the times
    /// k*time_step, k = 0..t_final/time_step.
    ReachSetDT reach(const ReachProblem &problem, const alg::DiscreteLinearization &alg) const;

    /// The step from R with the input set U (CORA's post and linReach): linearized at the center
    /// of R; the set after the step, reduced, and the error that was added.
    DiscreteStep post(const Zonotope &R, const std::optional<Zonotope> &U,
                      const alg::DiscreteLinearization &alg) const;

    /// The set of the outputs y = g(x, u) over the states R and the inputs U, and its error (CORA's
    /// outputSet).
    DiscreteStep output_set(const Zonotope &R, const std::optional<Zonotope> &U,
                            const alg::DiscreteLinearization &alg) const;

    /// The trajectories from the start points: x[k] at k*time_step, k = 0..t_final/time_step.
    DiscreteTrajectory simulate(const DiscreteSimulationProblem &problem) const;

  private:
    /// The checks of the sets of an operation: one zonotope of the states, one of the inputs.
    void validate_sets(const char *op, const Zonotope &R, const std::optional<Zonotope> &U) const;

    NonlinearSys state_, output_;
    double time_step_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
