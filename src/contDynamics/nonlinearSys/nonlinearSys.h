// nonlinearSys - the nonlinear system x' = f(x), as CORA's nonlinearSys (without inputs)
//
// The dynamics f is written on symbolic states (global/expr.h); the constructor differentiates it
// once, so the Jacobian and the Hessian that reach needs are exact. A single system and a single
// initial set: no batches. The tensors are on the backend of the initial set.
//
// Syntax:     NonlinearSys sys(f, n);   Reach R = sys.reach(X0, timeStep, tFinal, taylorTerms);
// Operations: reach, simulate, simulateRandom (one file each); the algorithm behind reach is in
//             private/
// See also:   contDynamics/linearSys/linearSys.h, global/expr.h

#pragma once

#include "contDynamics/linearSys/linearSys.h"
#include "global/expr.h"

#include <functional>
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

    /// f(x) at the point x, a column (n, 1).
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

    /// The numbers behind reach and simulate: f(x) and the Jacobian df_i/dx_j (row-major) at a
    /// point, f over a box, and the Hessian d2f_i/dx_j dx_k (row-major, i-th matrix first) over
    /// one.
    std::vector<double> values(const std::vector<double> &x) const;
    std::vector<double> jacobian(const std::vector<double> &x) const;
    std::vector<Range> enclosure(const std::vector<Range> &box) const;
    std::vector<Range> hessianEnclosure(const std::vector<Range> &box) const;

  private:
    int64_t n_;
    std::vector<Expr> f_, jacobian_, hessian_;
};

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
