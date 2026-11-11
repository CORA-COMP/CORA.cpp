// nonlinear_sys - the nonlinear system x' = f(x, u), as CORA's nonlinearSys
//
// The dynamics f is written on symbolic states and inputs (cora/global/expr.h); the constructor
// differentiates it, so the Jacobians and the Hessians that reach needs are exact (the third-order
// tensor is derived on its first use). A single system and a single initial set: no batches.
// Everything is computed on tensors, so gradients flow through reach and simulate with respect to
// the initial set; the decisions (which generators reduce keeps, how often the linearization
// error is enlarged) are made on the values.
//
// Syntax:     NonlinearSys sys(f, n);  NonlinearSys sys(g, n, m);  (f(x) or g(x, u))
//             ReachSet R = sys.reach({.initial_set = X0, .t_final = 1},
//             alg::Linearization{.time_step = 0.01});
// Operations: reach, init_reach and post (reach.cpp), linearize, simulate, simulate_random (one
//             file each); the constructor and the derivatives (nonlinear_sys.cpp)
// See also:   cora/cont_dynamics/linear_sys/linear_sys.h, cora/global/expr.h
//
// CORA: contDynamics/@nonlinearSys/nonlinearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/global/expr.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The dynamics and the results of its operations --------------------------------------------------

/// The right-hand side of x' = f(x): the n components, given the states x[0], ..., x[n-1].
using Dynamics = std::function<std::vector<Expr>(const std::vector<Expr> &x)>;

/// The right-hand side of x' = f(x, u), given the states x[0..n-1] and the inputs u[0..m-1].
using InputDynamics =
    std::function<std::vector<Expr>(const std::vector<Expr> &x, const std::vector<Expr> &u)>;

/// The Jacobians of f at a point (CORA's [A, B] = jacobian(x, u)): A (n, n) and B (n, m).
struct Jacobians {
    Tensor A, B;
};

/// The affine system of a step (CORA's linearize: linsys, linParams and linError): x' = A x + u
/// with u in U + u_const, expanded at (x_star, u_star).
struct Linearized {
    Tensor x_star, u_star, f0; ///< the expansion point and f there (CORA's p.x, p.u, f0)
    LinearSys sys;             ///< x' = A x with the input u in the state space (B = I)
    Zonotope U;                ///< B (U - u_star) without its center (CORA's linParams.U)
    Tensor u_const;            ///< f0 plus that center (CORA's linParams.uTrans)
};

/// One step of reach: the set over the step and at its end, and the linearization error that
/// the next step starts from (CORA's Rnext.ti, Rnext.tp.set and Rnext.tp.error).
struct NonlinearStep {
    Zonotope time_interval, time_point;
    Tensor error;
};

// The polynomial zonotopes of alg poly ------------------------------------------------------------

/// The sets of reach with alg::Polynomialization: the same fields as NonlinearStep.
struct PolyNonlinearStep {
    PolyZonotope time_interval, time_point;
    Tensor error;
};

/// What reach is asked with a polynomial zonotope as the initial set (CORA's params with R0 a
/// polyZonotope).
struct PolyReachProblem {
    std::optional<PolyZonotope> initial_set = std::nullopt; ///< the initial set X0, required
    std::optional<Zonotope> input_set = std::nullopt;       ///< the input set U; none: no input
    double t_final = 0;                                     ///< the time horizon, positive
};

/// The reachable sets of PolyReachProblem: time_interval[k] encloses the states over
/// [k*time_step, (k+1)*time_step], time_point[k] those at k*time_step.
struct PolyReachSet {
    std::vector<PolyZonotope> time_interval, time_point;
    double time_step = 0;
};

// The System --------------------------------------------------------------------------------------

class NonlinearSys {
  public:
    /// The system x' = f(x) of dimension n; f is called once, here.
    NonlinearSys(const Dynamics &f, int64_t n);

    /// The system x' = f(x, u) with n states and m inputs; f is called once, here.
    NonlinearSys(const InputDynamics &f, int64_t n, int64_t m);

    int64_t dim() const { return n_; }
    /// The components of f, as written on the symbolic states and inputs.
    const std::vector<Expr> &expressions() const { return f_; }
    int64_t input_dim() const { return m_; }

    /// f at the columns of x (n, N) and u (m, N) or (m, 1) (none: no inputs): (n, N).
    Tensor dynamics(const Tensor &x, const std::optional<Tensor> &u = std::nullopt) const;

    /// The Jacobians at the column x (n, 1) and u (m, 1).
    Jacobians jacobian(const Tensor &x, const std::optional<Tensor> &u = std::nullopt) const;

    /// f over the box x (bounds (n, 1)) and u (bounds (m, 1)): a range of shape (n, 1).
    Range enclosure(const Range &x, const std::optional<Range> &u = std::nullopt) const;

    /// The Hessians of the components over z = (x, u) at the point (CORA's hessian): n matrices
    /// (n + m, n + m).
    std::vector<Tensor> hessian(const Tensor &x,
                                const std::optional<Tensor> &u = std::nullopt) const;

    /// The Hessian d2f_i/dz_j dz_k over the box, row-major (i, j, k), each a range of shape (1, 1);
    /// an entry that is identically 0 is empty (CORA's hessian on intervals).
    std::vector<std::optional<Range>>
    hessian_enclosure(const Range &x, const std::optional<Range> &u = std::nullopt) const;

    /// The third derivatives d3f_i/dz_j dz_k dz_l over the box, row-major (i, j, k, l), empty where
    /// identically 0 (CORA's thirdOrderTensor on intervals: T{i,j}(k,l)).
    std::vector<std::optional<Range>>
    third_order_enclosure(const Range &x, const std::optional<Range> &u = std::nullopt) const;

    /// The affine system of a step from R with the input set U (CORA's linearize): expanded at
    /// the center of R moved by half a step of the flow, and at the center of U.
    Linearized linearize(const Zonotope &R, const std::optional<Zonotope> &U,
                         double time_step) const;

    /// The reachable sets from problem.initial_set with the input set problem.input_set.
    ReachSet reach(const ReachProblem &problem, const alg::NonlinearReach &alg) const;

    /// The first step of reach from R0 (CORA's initReach); post continues from it.
    NonlinearStep init_reach(const Zonotope &R0, const std::optional<Zonotope> &U,
                             const alg::NonlinearReach &alg) const;

    /// The step after `step` (CORA's post): as init_reach, then both sets are reduced.
    NonlinearStep post(const NonlinearStep &step, const std::optional<Zonotope> &U,
                       const alg::NonlinearReach &alg) const;

    /// The reachable sets from a polynomial zonotope with alg "poly" (CORA's reach with R0 a
    /// polyZonotope): tensor order 3, the Hessian at the expansion point as a polynomial zonotope.
    PolyReachSet reach(const PolyReachProblem &problem, const alg::PolyNonlinearReach &alg) const;

    /// The first step of that reach (CORA's initReach); post continues from it.
    PolyNonlinearStep init_reach(const PolyZonotope &R0, const std::optional<Zonotope> &U,
                                 const alg::Polynomialization &alg) const;

    /// The step after `step` (CORA's post): as init_reach, then both sets are reduced.
    PolyNonlinearStep post(const PolyNonlinearStep &step, const std::optional<Zonotope> &U,
                           const alg::Polynomialization &alg) const;

    /// linearize from the center (n, 1) of a set of any class, unchecked (the algorithms check).
    Linearized linearize_impl(const Tensor &center, const std::optional<Zonotope> &U,
                              double time_step) const;

    /// Trajectories from the points initial_state (n, N) at the times k*time_step, k = 0..steps,
    /// with the constant input `input` (m, N) or (m, 1).
    std::vector<Tensor> simulate(const SimulationProblem &problem,
                                 const alg::NonlinearSimulate &alg) const;

    /// simulate from `points` random points of initial_set (any set), each with a constant random
    /// input of input_set; rng makes them repeatable.
    std::vector<Tensor> simulate_random(const RandomSimulationProblem &problem,
                                        const alg::NonlinearSimulate &alg, int64_t points,
                                        Rng &rng) const;

  private:
    /// The variables of the expressions: the entries of x, then those of u.
    std::vector<Tensor> variables(const Tensor &x, const std::optional<Tensor> &u) const;
    std::vector<Range> variables(const Range &x, const std::optional<Range> &u) const;

    /// The checks of a reach problem's sets, for reach, linearize, init_reach and post.
    void validate_sets(const char *op, const Zonotope &R, const std::optional<Zonotope> &U) const;

    /// The checks of the input set U: only for a system with inputs, one zonotope of them.
    void validate_inputs(const char *op, const std::optional<Zonotope> &U) const;

    /// The checks of a polynomial zonotope R (one set of the dimension) and the input set U.
    void validate_poly_sets(const char *op, const PolyZonotope &R,
                            const std::optional<Zonotope> &U) const;

    /// The checks of a point x (n, 1) and an input u (m, 1) of the derivatives.
    void validate_point(const char *op, const Tensor &x, const std::optional<Tensor> &u) const;

    struct ThirdOrder; // the third-order tensor, derived once on first use (nonlinear_sys.cpp)

    int64_t n_, m_;
    std::vector<Expr> f_, jacobian_, hessian_; ///< row-major over (i, j) and (i, j, k)
    std::shared_ptr<ThirdOrder> third_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
