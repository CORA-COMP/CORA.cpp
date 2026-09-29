// linearSys - the linear system x' = A x, as CORA's linearSys (without inputs)
//
// A is a matrix (..., n, n); leading dimensions batch systems, and broadcast against those of
// the initial set, so one call covers a batch of systems, of sets, or of both.
//
// Syntax:     LinearSys sys(A);   Reach R = sys.reach(X0, timeStep, tFinal, taylorTerms);
// Operations: reach, simulate, simulateRandom, correctionMatrixState (one file each); the
//             algorithms behind reach are in private/
// See also:   contSet/zonotope/zonotope.h, specification/specification.h

#pragma once

#include "contSet/zonotope/zonotope.h"

#include <vector>

namespace cora::ct {

/// Which algorithm computes the reachable sets (CORA's linAlg).
enum class Algorithm {
    Standard,      ///< "standard": encloses the set of every step
    WrappingFree,  ///< "wrapping-free": encloses the first step once and maps it forward
};

/// The reachable sets: timeInt[k] over [k*timeStep, (k+1)*timeStep], timePoint[k] at k*timeStep.
struct Reach {
    std::vector<Zonotope> timeInt, timePoint;
};

class LinearSys {
  public:
    /// The system matrix A (..., n, n).
    explicit LinearSys(Tensor A) : A_(std::move(A)) {}

    const Tensor &A() const { return A_; }

    /// The reachable sets from X0 over ceil(tFinal / timeStep) steps; taylorTerms is the order
    /// of the Taylor series behind the curvature enlargement.
    Reach reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms,
                Algorithm linAlg = Algorithm::Standard) const;

    /// Trajectories from the points x0 (..., n, N) at the times k*timeStep, k = 0..steps:
    /// x[k] = e^{A k timeStep} x0, exact for a linear system.
    std::vector<Tensor> simulate(const Tensor &x0, double timeStep, double tFinal) const;

    /// simulate from N random points of X0 (any set); rng makes the points repeatable.
    std::vector<Tensor> simulateRandom(const ContSet &X0, int64_t N, double timeStep, double tFinal,
                                       Rng &rng) const;

    /// The interval matrix F(A, timeStep, taylorTerms) that encloses the curvature of the
    /// trajectories between two time points when multiplied with the set at the first.
    Interval correctionMatrixState(double timeStep, int taylorTerms) const;

  private:
    Tensor A_;
};

} // namespace cora::ct
