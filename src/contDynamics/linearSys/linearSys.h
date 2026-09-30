// linearSys - the linear system x' = A x + B u, y = C x, as CORA's linearSys
//
// A is a matrix (..., n, n); leading dimensions batch systems, and broadcast against those of
// the initial set, so one call covers a batch of systems, of sets, or of both. B (n, m) and C (p, n)
// are optional single matrices; without C the output is the state.
//
// Syntax:     LinearSys sys(A);   Reach R = sys.reach(X0, timeStep, tFinal, taylorTerms);
//             LinearSys sys(A, B, C);   VerifyResult r = sys.verify(params, alg, specs);
// Operations: reach, simulate, simulateRandom, correctionMatrixState, correctionMatrixInput,
//             verify (one file each); the algorithms behind reach and verify are in private/
// See also:   contSet/zonotope/zonotope.h, specification/specification.h

#pragma once

#include "contSet/zonotope/zonotope.h"
#include "specification/specification.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Which algorithm computes the reachable sets (CORA's linAlg).
enum class Algorithm {
    Standard,      ///< "standard": encloses the set of every step
    WrappingFree,  ///< "wrapping-free": encloses the first step once and maps it forward
};

/// The reachable sets: timeInt[k] over [k*timeStep, (k+1)*timeStep], timePoint[k] at k*timeStep.
struct Reach {
    std::vector<Zonotope> timeInt, timePoint;
};

/// Which algorithm verify runs (CORA's options.verifyAlg).
enum class VerifyAlg {
    SupportFunc,  ///< "reachavoid:supportFunc": support function of the affine solution, adaptive step
    Zonotope,     ///< "reachavoid:zonotope": zonotope reachable sets, adaptive step
};

/// What verify is given (CORA's params): the initial set, the set of the input u(t) in R^m, and
/// the time horizon. R0 lives in the state space, U in the input space of B.
struct VerifyParams {
    Zonotope R0;
    Zonotope U;
    double tFinal;
};

/// A trajectory that violates a specification: the state at time 0 and the time it is violated.
struct Falsification {
    Tensor x0;
    double tFinal;
};

/// The outcome of verify (CORA's res, fals and savedata).
struct VerifyResult {
    bool verified = false;
    double tComp = 0;         ///< seconds spent, as CORA's savedata.tComp
    int iterations = 0;       ///< passes of the adaptive loop
    double timeStep = 0;      ///< the step size of the last pass
    int nrSteps = 0;          ///< the number of steps of the last pass
    std::optional<Falsification> fals;
};

class LinearSys {
  public:
    /// The autonomous system x' = A x, A (..., n, n).
    explicit LinearSys(Tensor A) : A_(std::move(A)) {}

    /// x' = A x + B u with the output y = C x (C empty: y = x); A is a single matrix here.
    LinearSys(Tensor A, Tensor B, std::optional<Tensor> C = std::nullopt)
        : A_(std::move(A)), B_(std::move(B)), C_(std::move(C)) {}

    const Tensor &A() const { return A_; }
    const std::optional<Tensor> &B() const { return B_; }
    const std::optional<Tensor> &C() const { return C_; }

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

    /// The interval matrix G(A, timeStep, taylorTerms) that encloses the curvature of the input
    /// solution: multiplied with u, it bounds what the input does between two time points.
    Interval correctionMatrixInput(double timeStep, int taylorTerms) const;

    /// Whether every specification holds for the outputs y = C x (+ B u reached from R0 and U)
    /// up to params.tFinal. `specs` are over the output space.
    VerifyResult verify(const VerifyParams &params, VerifyAlg alg,
                        const std::vector<Specification> &specs) const;

  private:
    Tensor A_;
    std::optional<Tensor> B_, C_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
