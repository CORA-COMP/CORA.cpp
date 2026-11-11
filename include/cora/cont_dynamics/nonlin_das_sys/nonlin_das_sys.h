// nonlin_das_sys - x' = f(x, y, u), 0 = g(x, y, u), as CORA's nonlinDASys
//
// f and g are written on symbolic states x, algebraic variables y and inputs u (global/expr.h);
// the constructor differentiates them. The system has index 1: dg/dy is invertible along the
// reachable set. reach linearizes at the center of the set, eliminates y and bounds the
// linearization error by the Lagrange remainder of the second order. A decomposition into
// subsystems (CORA's options.index, [Althoff, TPS 2014]) evaluates the remainder per subsystem on
// its own equations. The sign of the algebraic error and the coupling of the subsystems are
// corrected as CORALean does (docs/cora-differences.md).
//
// Syntax:     NonlinDASys sys(f, g, n, m, p);   (f and g on the symbolic x, y and u)
//             DaeReachSet R = sys.reach({.initial_set = X0, .y0_guess = y0, .t_final = 1},
//             alg::DaeLinearization{.time_step = 0.01});
// Operations: reach, init_reach and post (reach.cpp), consistent_initial_state, steady_state,
//             linearize, simulate (one file each); the constructor and the derivatives
//             (nonlin_das_sys.cpp)
// See also:   cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h, cora/global/expr.h
//
// CORA: contDynamics/@nonlinDASys/nonlinDASys.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/expr.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The dynamics and the results of its operations --------------------------------------------------

/// The right-hand side of f or g, given the symbolic x[0..n-1], y[0..p-1] and u[0..m-1].
using DaeDynamics = std::function<std::vector<Expr>(
    const std::vector<Expr> &x, const std::vector<Expr> &y, const std::vector<Expr> &u)>;

/// The Jacobians at a point (CORA's [A, B, C, D, E, F] = jacobian(x, y, u)): A = df/dx, B = df/du,
/// C = df/dy, D = dg/dx, E = dg/du and F = dg/dy.
struct DaeJacobians {
    Tensor A, B, C, D, E, F;
};

/// The Hessians over the box z = (x, y, u): the second derivatives of the n rows of f, then of the
/// p rows of g, each row-major (i, j, k) over the d = n + p + m variables; an entry that is
/// identically 0 is empty (CORA's hessian on intervals, hessianTensorInt).
struct DaeHessians {
    std::vector<std::optional<Range>> f, g;
};

/// The affine system of a step with the algebraic variables eliminated (CORA's priv_linearize and
/// the fields of its linError): x' = A_lin x + u with the input in U + u_const, expanded at
/// (x_star, y_star, u_star).
struct DaeLinearized {
    Tensor x_star, y_star, u_star; ///< the expansion point (CORA's p.x, p.y, p.u)
    Tensor g0;                     ///< g at the point (CORA's f0_con)
    Tensor D, E, F_inv, CF_inv;    ///< dg/dx, dg/du, pinv(dg/dy) and df/dy * pinv(dg/dy)
    LinearSys sys;                 ///< x' = A_lin x with the input in the state space (B = I)
    Zonotope U;                    ///< B_lin (U - u_star), centered (CORA's linParams.U)
    Tensor u_const;                ///< f at the point minus C F^-1 g there (CORA's f0)
};

/// One step of reach: the sets over the step and at its end, the algebraic variables over the
/// step, and the linearization error the next step starts from (CORA's Rnext.ti, .tp.set, .y,
/// .tp.error_x and .tp.error_y).
struct DaeStep {
    Zonotope time_interval, time_point, algebraic;
    Tensor error_x, error_y;
};

/// The reachable sets: time_interval[k] and algebraic[k] over [t_start + k*time_step, t_start +
/// (k+1)*time_step], time_point[k] at t_start + k*time_step.
struct DaeReachSet {
    std::vector<Zonotope> time_interval, time_point, algebraic;
    double time_step = 0;
};

/// What simulate returns: the states x (n, K+1) and the algebraic variables y (p, K+1) at the
/// sample times.
struct DaeTrajectory {
    Tensor x, y;
};

struct DaeReachProblem; // after the class: it holds a decomposition into systems

// The System --------------------------------------------------------------------------------------

class NonlinDASys {
  public:
    /// The system of n states, m inputs and p algebraic variables; f and g are called once, here.
    NonlinDASys(const DaeDynamics &f, const DaeDynamics &g, int64_t n, int64_t m, int64_t p);

    int64_t dim() const { return n_; }
    int64_t input_dim() const { return m_; }
    int64_t constraint_dim() const { return p_; }

    /// f and g at the columns of x (n, N), y (p, N) and u (m, N) or (m, 1): (n, N) and (p, N).
    Tensor dynamics(const Tensor &x, const Tensor &y, const Tensor &u) const;
    Tensor constraints(const Tensor &x, const Tensor &y, const Tensor &u) const;

    /// The Jacobians at the columns x (n, 1), y (p, 1) and u (m, 1).
    DaeJacobians jacobian(const Tensor &x, const Tensor &y, const Tensor &u) const;

    /// The Hessians of f and g over the box x, y, u (CORA's hessian after setHessian('int')); the
    /// symbolic second derivatives are derived by the first call that needs them.
    DaeHessians hessian_enclosure(const Range &x, const Range &y, const Range &u) const;

    /// The algebraic variables y with g(x0, y, u0) = 0 by Newton's method from y0 (CORA's
    /// consistentInitialState): the end point is the last Newton step, taken when its 2-norm is
    /// below 1e-10.
    Tensor consistent_initial_state(const Tensor &x0, const Tensor &y0, const Tensor &u0) const;

    /// The state and the algebraic variables with f = 0 and g = 0 at u0 by Newton's method on the
    /// stacked system from (x0, y0) (CORA's steadyState).
    std::pair<Tensor, Tensor> steady_state(const Tensor &x0, const Tensor &y0,
                                           const Tensor &u0) const;

    /// The affine system of a step from the point (x0, y0), the center of the set: expanded half a
    /// step along the flow with y made consistent (CORA's priv_linearize).
    DaeLinearized linearize(const Tensor &x0, const Tensor &y0, const std::optional<Zonotope> &U,
                            const Tensor &u_trans, double time_step) const;

    /// The reachable sets from problem.initial_set with the piecewise constant inputs
    /// problem.u_trans (one column per step) and the input set problem.input_set.
    DaeReachSet reach(const DaeReachProblem &problem, const alg::DaeReach &alg) const;

    /// The first step of reach from R0 (CORA's initReach); post continues from it.
    DaeStep init_reach(const DaeReachProblem &problem, const alg::DaeReach &alg) const;

    /// The step after `step` (CORA's post) with the input u_trans: as init_reach, then both sets
    /// are reduced.
    DaeStep post(const DaeStep &step, const DaeReachProblem &problem, const Tensor &u_trans,
                 const alg::DaeReach &alg) const;

    /// A trajectory from the state x0 with the algebraic variables made consistent from y_guess,
    /// under the constant input u: the classical Runge-Kutta method, y solved at every stage.
    DaeTrajectory simulate(const Tensor &x0, const Tensor &y_guess, const Tensor &u, double t_final,
                           const alg::NonlinearSimulate &alg) const;

  private:
    /// The variables of the expressions: the entries of x, then y, then u.
    std::vector<Tensor> variables(const Tensor &x, const Tensor &y, const Tensor &u) const;
    std::vector<Range> variables(const Range &x, const Range &y, const Range &u) const;

    /// The algebraic variables with g(x, y, u) = 0 by Newton's method from y, unchecked; the last
    /// step is added if add_last (consistentInitialState), else not (priv_linearize's helper).
    Tensor newton_algebraic(const Tensor &x, const Tensor &y, const Tensor &u, bool add_last) const;

    /// The check of a state x (n, 1) and a guess y (p, 1) of the operation `op`.
    void validate_state(const char *op, const Tensor &x, const Tensor &y) const;

    struct Second; // the second derivatives, derived once on first use (nonlin_das_sys.cpp)

    int64_t n_, m_, p_;
    std::vector<Expr> f_, g_, jacobian_; ///< the rows of f then g; jacobian_ row-major (i, j)
    std::shared_ptr<Second> second_;
};

// The decomposition -----------------------------------------------------------------------------

/// One subsystem of a decomposition (CORA's index{iSys} and the subsystem model): its equations
/// on the states and constraints it owns, the other variables it reads as inputs. X, Y select its
/// states and constraints; Uy, Uu and Uv its inputs from y, from u and from the voltages v_gen.
struct DaeSubsystem {
    NonlinDASys sys;
    Tensor X, Y, Uv, Uy, Uu;
};

/// A decomposition of the system into subsystems whose equations are the rows of f and g they own
/// (CORA's options.index, subsystemHessian and Vgen): the Lagrange remainder of the whole system is
/// the sum of the remainders of its parts.
struct DaeDecomposition {
    std::vector<DaeSubsystem> subsystems;
    Tensor v_gen; ///< the constant voltages (nv, 1) the inputs of Uv read (CORA's options.Vgen)
};

/// What reach is asked (CORA's params of nonlinDASys).
struct DaeReachProblem {
    std::optional<Zonotope> initial_set = std::nullopt; ///< the initial set X0 (R0), required
    std::optional<Zonotope> input_set = std::nullopt;   ///< the set U of the input; none: {0}
    std::optional<Tensor> u_trans = std::nullopt;       ///< the constant inputs (m, K) or (m, 1)
    std::optional<Tensor> y0_guess = std::nullopt; ///< the guess of y at the start (p, 1), required
    double t_start = 0;                            ///< the start time (CORA's tStart)
    double t_final = 0;                            ///< the end time, above t_start (CORA's tFinal)
    std::optional<DaeDecomposition> decomposition = std::nullopt; ///< none: the whole system

    /// The fields Python fills by keyword; the decomposition is built in C++ only.
    static constexpr auto fields() {
        return std::make_tuple(field("initial_set", &DaeReachProblem::initial_set),
                               field("input_set", &DaeReachProblem::input_set),
                               field("u_trans", &DaeReachProblem::u_trans),
                               field("y0_guess", &DaeReachProblem::y0_guess),
                               field("t_start", &DaeReachProblem::t_start),
                               field("t_final", &DaeReachProblem::t_final));
    }
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
