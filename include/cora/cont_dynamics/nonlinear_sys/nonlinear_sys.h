// nonlinear_sys - the nonlinear system x' = f(x), as CORA's nonlinearSys (without inputs)
//
// The dynamics f is written on symbolic states (cora/global/expr.h); the constructor differentiates
// it once, so the Jacobian and the Hessian that reach needs are exact. A single system and a single
// initial set: no batches. Everything is computed on tensors, so gradients flow through reach and
// simulate with respect to the initial set; the decisions (which generators reduce keeps, when the
// enclosure of a step has converged) are made on the values.
//
// Syntax:     NonlinearSys sys(f, n);
//             ReachSet R = sys.reach({.initial_set = X0, .t_final = 1},
//             alg::Linearization{.time_step = 0.01});
// Operations: reach, simulate, simulate_random (one file each); the constructor, dynamics,
//             jacobian, enclosure and hessian_enclosure (nonlinear_sys.cpp)
// See also:   cora/cont_dynamics/linear_sys/linear_sys.h, cora/global/expr.h
//
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/expr.h"

#include <functional>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The right-hand side of x' = f(x): the n components, given the states x[0], ..., x[n-1].
using Dynamics = std::function<std::vector<Expr>(const std::vector<Expr> &x)>;

class NonlinearSys {
  public:
    /// The system of dimension n with the right-hand side f; f is called once, here.
    NonlinearSys(const Dynamics &f, int64_t n);

    int64_t dim() const { return n_; }

    /// f at the columns of x (n, N): (n, N).
    Tensor dynamics(const Tensor &x) const;

    /// The reachable sets from problem.initial_set (no input set) up to t_final.
    ReachSet reach(const ReachProblem &problem, const alg::NonlinearReach &alg) const;

    /// Trajectories from the points initial_state (n, N) at the times k*time_step, k = 0..steps.
    std::vector<Tensor> simulate(const SimulationProblem &problem,
                                 const alg::NonlinearSimulate &alg) const;

    /// simulate from `points` random points of initial_set (any set); rng makes them repeatable.
    std::vector<Tensor> simulate_random(const RandomSimulationProblem &problem,
                                        const alg::NonlinearSimulate &alg, int64_t points,
                                        Rng &rng) const;

    /// The Jacobian df_i/dx_j at the column x (n, 1): (n, n).
    Tensor jacobian(const Tensor &x) const;

    /// f over the box (bounds of shape (n, 1)): a range of shape (n, 1).
    Range enclosure(const Range &box) const;

    /// The Hessian d2f_i/dx_j dx_k over the box, row-major (i, j, k), each a range of shape (1, 1);
    /// an entry that is identically 0 is empty.
    std::vector<std::optional<Range>> hessian_enclosure(const Range &box) const;

  private:
    int64_t n_;
    std::vector<Expr> f_, jacobian_, hessian_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
