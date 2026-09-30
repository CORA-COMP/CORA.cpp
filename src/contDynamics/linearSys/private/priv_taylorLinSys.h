// priv_taylorLinSys - cache of the Taylor quantities of a linear system, as CORA's taylorLinSys
//
// Holds what the adaptive algorithms recompute otherwise: the powers of A and their positive and
// negative parts, dt^i / i!, e^{A dt} and the correction matrices F, G per time step size (a
// step size counts as used before if it matches up to 1e-10).
//
// Syntax:   TaylorLinSys taylor(A);   Tensor eAdt = taylor.eAdt(0.1);   auto F = taylor.F(0.1);
// Inputs:   A - system matrix (n, n)
// Outputs:  the cached quantities; F and G are empty if their Taylor sums do not converge
// See also: priv_reach_adaptive, priv_verifyRA_zonotope

#pragma once

#include "contSet/interval/interval.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class TaylorLinSys {
  public:
    explicit TaylorLinSys(Tensor A);

    const Tensor &A() const { return A_; }

    /// A^i for i >= 1, and its positive and negative parts.
    Tensor Apower(int i);
    Tensor Apos(int i);
    Tensor Aneg(int i);

    /// dt^i / i! for i >= 1.
    double dtoverfac(double dt, int i);

    /// e^{A dt}, computed and cached if the step size is new.
    Tensor eAdt(double dt);

    /// e^{A dt} if it is cached; insertEAdt caches a value computed elsewhere.
    std::optional<Tensor> readEAdt(double dt) const;
    void insertEAdt(double dt, Tensor value);

    /// A^-1, empty if A is singular.
    std::optional<Tensor> Ainv();

    /// The correction matrix of the state (Taylor sum to floating-point precision) as an
    /// interval matrix; empty if the sum has not converged after 75 terms.
    std::optional<Interval> F(double dt);

    /// The same for the input.
    std::optional<Interval> G(double dt);

  private:
    struct PerStep {
        double dt;
        std::vector<double> dtoverfac;
        std::optional<Tensor> eAdt;
        std::optional<Interval> F, G;
    };

    /// The cache entry for dt, created if needed; or null when only reading.
    PerStep &entry(double dt);
    const PerStep *find(double dt) const;

    Tensor A_;
    std::vector<Tensor> Apower_, Apos_, Aneg_;
    std::optional<Tensor> Ainv_;
    bool AinvDone_ = false;
    std::vector<PerStep> steps_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
