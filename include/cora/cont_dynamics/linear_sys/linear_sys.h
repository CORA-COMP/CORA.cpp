// linear_sys - the linear system x' = A x + B u, y = C x, as CORA's linearSys
//
// A is a matrix (..., n, n); leading dimensions batch systems, and broadcast against those of
// the initial set, so one call covers a batch of systems, of sets, or of both. B (n, m) and C
// (p, n) are optional single matrices; without C the output is the state. An operation takes
// the problem (CORA's params) and the algorithm (CORA's options) apart.
//
// Syntax:     LinearSys sys(A);   LinearSys sys(A, B, C);
//             ReachSet R = sys.reach({.initial_set = X0, .t_final = 2}, alg::Standard{.time_step =
//             0.1}); VerifyResult r = sys.verify({.initial_set = X0, .input_set = U, .t_final = 2},
//             specs,
//                                         alg::ReachAvoidZonotope{});
// Operations: reach, output_set, simulate, simulate_random, verify (one file each); the building
//             blocks of reach correction_matrix_state, correction_matrix_input, expm_remainder,
//             input_solution; validate (the checks of a problem); the algorithms behind reach and
//             verify are in private/; the constructors check their input (linear_sys.cpp)
// See also:   cora/cont_dynamics/algorithms.h, cora/cont_dynamics/cont_dynamics.h
//
// CORA: contDynamics/@linearSys/linearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/specification/specification.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Results -----------------------------------------------------------------------------------------

/// A trajectory that violates a specification: the state at time 0 and the time it is violated.
struct Falsification {
    Tensor x0;
    double time;
};

/// The outcome of verify (CORA's res, fals and savedata).
struct VerifyResult {
    bool verified = false;
    double computation_time = 0; ///< seconds spent, as CORA's savedata.tComp
    int iterations = 0;          ///< passes of the adaptive loop
    double time_step = 0;        ///< the step size of the last pass
    int num_steps = 0;           ///< the number of steps of the last pass
    std::optional<Falsification> falsification;
};

/// What the input u(t) in U adds over one step (CORA's particularSolution_timeVarying and
/// _constant, and the curvature of the input). With U = c + U0 (U0 centered): the constant part
/// is B c and the time-varying part B U0.
struct InputSolution {
    Zonotope PU;     ///< time-varying solution at t = time_step, centered at the origin
    Tensor Pu;       ///< constant-input solution at t = time_step (n, 1)
    Zonotope Cinput; ///< curvature enlargement of the constant input
};

// The System --------------------------------------------------------------------------------------

class LinearSys {
  public:
    /// The autonomous system x' = A x, A (..., n, n).
    explicit LinearSys(Tensor A);

    /// x' = A x + B u with the output y = C x (C empty: y = x); A is a single matrix here.
    LinearSys(Tensor A, Tensor B, std::optional<Tensor> C = std::nullopt);

    const Tensor &A() const noexcept { return A_; }
    const std::optional<Tensor> &B() const noexcept { return B_; }
    const std::optional<Tensor> &C() const noexcept { return C_; }

    /// The number of states n.
    int64_t dim() const { return A_.shape().back(); }

    /// The reachable sets from problem.initial_set over ceil(t_final / time_step) steps, for every
    /// u(t) in problem.input_set (needs B) if there is one.
    ReachSet reach(const ReachProblem &problem, const alg::LinearReach &alg) const;

    /// The output sets y = C x of every set of R (R itself without C), as CORA's outputSet.
    ReachSet output_set(const ReachSet &R) const;

    /// Trajectories from the points initial_state (..., n, N) at the times k*time_step,
    /// k = 0..steps: exact for a linear system, with the constant inputs (needs B) if given.
    std::vector<Tensor> simulate(const SimulationProblem &problem,
                                 const alg::LinearSimulate &alg) const;

    /// simulate from `points` random points of initial_set (any set), with as many random
    /// constant inputs of input_set if given; rng makes the points repeatable.
    std::vector<Tensor> simulate_random(const RandomSimulationProblem &problem,
                                        const alg::LinearSimulate &alg, int64_t points,
                                        Rng &rng) const;

    /// The interval matrix F(A, time_step, taylor_terms) that encloses the curvature of the
    /// trajectories between two time points when multiplied with the set at the first.
    Interval correction_matrix_state(double time_step, int taylor_terms) const;

    /// The interval matrix G(A, time_step, taylor_terms) that encloses the curvature of the input
    /// solution: multiplied with u, it bounds what the input does between two time points.
    Interval correction_matrix_input(double time_step, int taylor_terms) const;

    /// The error box [-W, W] of the series sum_{i<=taylor_terms} |A|^i dt^i / i! against
    /// e^{|A| dt}; multiplied with dt it bounds the truncated series of the input.
    Interval expm_remainder(double time_step, int taylor_terms) const;

    /// What the input u(t) in U (needs B) adds over one step of reach.
    InputSolution input_solution(const Zonotope &U, double time_step, int taylor_terms) const;

    /// Checks a problem against the system, as reach does: the initial set, the input set and
    /// t_final. Algorithms registered from Python get the same checks.
    void validate(const ReachProblem &problem) const;

    /// Whether every specification holds for the outputs y = C x (+ B u reached from the initial
    /// set and the input set) up to t_final. `specs` are over the output space.
    VerifyResult verify(const ReachProblem &problem, const std::vector<Specification> &specs,
                        const alg::LinearVerify &alg) const;

  private:
    Tensor A_;
    std::optional<Tensor> B_, C_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
