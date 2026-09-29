// nonlinearSys - the nonlinear system x' = f(x), as CORA's nonlinearSys (without inputs)
//
// The dynamics f is written on symbolic states (global/expr.h); the constructor differentiates it
// once, so the Jacobian and the Hessian that reach needs are exact. A single system and a single
// initial set: no batches. Everything is computed on tensors, so gradients flow through reach and
// simulate with respect to the initial set; the decisions (which generators reduce keeps, when the
// enclosure of a step has converged) are made on the values.
//
// Syntax:     NonlinearSys sys(f, n);   Reach R = sys.reach(X0, timeStep, tFinal, taylorTerms);
// Operations: reach, simulate, simulateRandom (one file each); the algorithm behind reach is in
//             private/
// See also:   contDynamics/linearSys/linearSys.h, global/expr.h

#pragma once

#include "contDynamics/linearSys/linearSys.h"
#include "global/expr.h"

#include <functional>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

/// The right-hand side of x' = f(x): the n components, given the states x[0], ..., x[n-1].
using Dynamics = std::function<std::vector<Expr>(const std::vector<Expr> &x)>;

class NonlinearSys {
  public:
    /// The system of dimension n with the right-hand side f; f is called once, here.
    NonlinearSys(const Dynamics &f, int64_t n);

    int64_t dim() const { return n_; }

    /// f at the columns of x (n, N): (n, N).
    Tensor dynamics(const Tensor &x) const;

    /// The reachable sets from X0 (CORA's algorithm "lin"): every step linearizes f at the center
    /// of the current set and bounds the linearization error by the Lagrange remainder over an
    /// enclosure of the step. taylorTerms is the order of the Taylor series of the linearized
    /// system; zonotopeOrder limits the generators of a set to zonotopeOrder * n.
    Reach reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms = 4,
                int zonotopeOrder = 50) const;

    /// Trajectories from the points x0 (n, N) at the times k*timeStep, k = 0..steps: the classical
    /// Runge-Kutta method with steps of at most 1e-3.
    std::vector<Tensor> simulate(const Tensor &x0, double timeStep, double tFinal) const;

    /// simulate from N random points of X0 (any set); rng makes the points repeatable.
    std::vector<Tensor> simulateRandom(const ContSet &X0, int64_t N, double timeStep, double tFinal,
                                       Rng &rng) const;

    /// The Jacobian df_i/dx_j at the column x (n, 1): (n, n).
    Tensor jacobian(const Tensor &x) const;

    /// f over the box (bounds of shape (n, 1)): a range of shape (n, 1).
    Range enclosure(const Range &box) const;

    /// The Hessian d2f_i/dx_j dx_k over the box, row-major (i, j, k), each a range of shape (1, 1);
    /// an entry that is identically 0 is empty.
    std::vector<std::optional<Range>> hessianEnclosure(const Range &box) const;

  private:
    int64_t n_;
    std::vector<Expr> f_, jacobian_, hessian_;
};

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
