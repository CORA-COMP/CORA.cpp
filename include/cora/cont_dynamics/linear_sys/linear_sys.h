// linear_sys - the linear system x' = A x + B u, y = C x, as CORA's linearSys
//
// A is a matrix (..., n, n); leading dimensions batch systems, and broadcast against those of
// the initial set, so one call covers a batch of systems, of sets, or of both. B (n, m) and C
// (p, n) are optional single matrices; without C the output is the state. A may be sparse
// (SparseMatrix): alg::Krylov uses its products, every other operation its dense form. An
// operation takes the problem (CORA's params) and the algorithm (CORA's options) apart.
//
// Syntax:     LinearSys sys(A);   LinearSys sys(A, B, C);
//             ReachSet R = sys.reach({.initial_set = X0, .t_final = 2}, alg::Standard{});
//             VerifyResult r = sys.verify(problem, specs, alg::ReachAvoidZonotope{});
// Operations: reach, output_set, simulate, simulate_random, verify (one file each); the building
//             blocks of reach correction_matrix_state, correction_matrix_input, expm_remainder,
//             input_solution, canonical_form, homogeneous_solution, particular_solution_*,
//             affine_solution, one_step, init_reach, post, error_solution_adaptive; validate (the
//             checks of a problem); equality, printing, generate_random, nonlinear_sys; the
//             algorithms behind reach and verify are in private/; the constructors check their
//             input (linear_sys.cpp)
// See also:   cora/cont_dynamics/algorithms.h, cora/cont_dynamics/cont_dynamics.h
//
// CORA: contDynamics/@linearSys/linearSys.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/sparse_matrix.h"
#include "cora/specification/specification.h"

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
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

/// What canonical_form returns (CORA's linsys_, U_, u_): the system with B = I, the centered input
/// set B (U - c) and the constant input B (c + u_vec) (n, steps). CORA's V_, v_ are zero here.
struct CanonicalForm;

/// The homogeneous solution of one step (CORA's Htp, Hti, C_state): e^{A dt} X at the end of the
/// step, the enclosure of the whole step, and the curvature enlargement F X it contains.
struct HomogeneousSolution {
    Zonotope time_point, time_interval, state_curvature;
};

/// The solution of one step for the constant input u (CORA's Ptp, C_input, Pti): the set at the
/// end of the step, the curvature enlargement G u, and the enclosure conv(0, Ptp) + G u.
struct ConstantSolution {
    Zonotope time_point, input_curvature, time_interval;
};

/// The solution of one step with a constant input (CORA's Htp, Pu, Hti, C_state, C_input): the
/// set at the end, the part of the input, the enclosure of the step and the two curvatures.
struct AffineSolution {
    Zonotope time_point, constant, time_interval, state_curvature, input_curvature;
};

/// One step of reach (CORA's Rtp, Rti, Htp, Hti, PU, Pu, C_state, C_input): the reachable sets
/// at the end and over the step, then the homogeneous solutions (time point, interval), the
/// time-varying and the constant part of the input, and the two curvatures.
struct OneStep {
    Zonotope time_point, time_interval, homogeneous_time_point, homogeneous_time_interval,
        time_varying, constant, state_curvature, input_curvature;
};

/// What trajectory_error returns (CORA's E_err, c_tilde, d_tilde): the set that encloses the error
/// of the linearized trajectory, and the boxes of the coefficients and the offset it gives n' x.
struct TrajectoryError {
    Zonotope error;
    Interval c_tilde, d_tilde;
};

/// What a step carries to the next (CORA's options.Rhom, Rhom_tp, Raux, Rpar, Rtrans): the
/// homogeneous sets over the step and at its end, the first input solution (aux), the sum of the
/// input solutions so far (par; a box if wrapping-free) and the constant input (trans).
struct ReachState {
    Zonotope homogeneous, homogeneous_time_point, aux, par, trans;
};

/// The sets of one step of init_reach or post with the state for the next (CORA's Rfirst/Rnext
/// .ti and .tp, and the options they update).
struct ReachStep {
    Zonotope time_interval, time_point;
    ReachState state;
};

class NonlinearSys;

// The System --------------------------------------------------------------------------------------

class LinearSys {
  public:
    /// The autonomous system x' = A x, A (..., n, n).
    explicit LinearSys(Tensor A);

    /// x' = A x + B u with the output y = C x (C empty: y = x); A is a single matrix here.
    LinearSys(Tensor A, Tensor B, std::optional<Tensor> C = std::nullopt);

    /// The same with a sparse A (n, n): reach with alg::Krylov uses only its products with
    /// vectors; any other operation forms the dense A (like B) once, when it first needs it.
    LinearSys(SparseMatrix A, Tensor B, std::optional<Tensor> C = std::nullopt);

    /// The dense A (formed on first use for a sparse system).
    const Tensor &A() const;
    /// The sparse A of a system built from one, else null.
    const SparseMatrix *A_sparse() const noexcept { return A_sparse_.get(); }
    const std::optional<Tensor> &B() const noexcept { return B_; }
    const std::optional<Tensor> &C() const noexcept { return C_; }

    /// The number of states n.
    int64_t dim() const { return A_sparse_ ? A_sparse_->rows() : A_.shape().back(); }

    /// The reachable sets from problem.initial_set over ceil(t_final / time_step) steps, for every
    /// u(t) in problem.input_set (needs B) if there is one.
    ReachSet reach(const ReachProblem &problem, const alg::LinearReach &alg) const;

    /// The output sets y = C x (the states without C) by CORA's Krylov subspace algorithm, which
    /// never forms a set of the state space; output_set must not be applied to them again.
    ReachSet reach(const ReachProblem &problem, const alg::Krylov &alg) const;

    /// As reach with alg::Krylov, but each step's time-interval set is passed to on_time_interval
    /// and nothing is stored (HEAT04's 1387 outputs over 8000 steps would take 250 GB); the
    /// returned ReachSet has no sets.
    ReachSet reach(const ReachProblem &problem, const alg::Krylov &alg,
                   const std::function<void(const Zonotope &)> &on_time_interval) const;

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

    // The building blocks of reach ----------------------------------------------------------------

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

    // The steps of reach -------------------------------------------------------------------------

    /// The system with B = I, the centered input set B (U - c) and the constant input B (c + u_vec)
    /// (CORA's canonicalForm); u_vec (m, steps) is one constant input per step (default 0).
    CanonicalForm canonical_form(const Zonotope &U,
                                 const std::optional<Tensor> &u_vec = std::nullopt) const;

    /// Whether B = I and C = I (square, equal entries; without B, never; without C, as C = I),
    /// the form canonical_form produces (CORA's isCanonicalForm).
    bool is_canonical_form() const;

    /// e^{A dt} X, the enclosure of the step and F X (CORA's homogeneousSolution); CORA's blocks
    /// argument is what alg::Decomposition uses.
    HomogeneousSolution homogeneous_solution(const Zonotope &X, double time_step,
                                             int taylor_terms) const;

    /// The solution at the end of the step of the input u(t) in U, whose terms A^i are kept
    /// apart (CORA's particularSolution_timeVarying); U is a set of the state space.
    Zonotope particular_solution_time_varying(const Zonotope &U, double time_step,
                                              int taylor_terms) const;

    /// The solution of one step for a constant input in U (CORA's particularSolution_constant):
    /// A^-1 (e^{A dt} - I) U if A is invertible, else the truncated series. Not a batch.
    ConstantSolution particular_solution_constant(const Zonotope &U, double time_step,
                                                  int taylor_terms) const;

    /// The homogeneous solution plus the constant input u (n, 1) (CORA's affineSolution).
    AffineSolution affine_solution(const Zonotope &X, const Tensor &u, double time_step,
                                   int taylor_terms) const;

    /// One step of reach from X with the input set U and the constant input u (n, 1) (CORA's
    /// oneStep). CORA's blocks argument is what alg::Decomposition uses; here the state is one
    /// block. Not a batch of systems.
    OneStep one_step(const Zonotope &X, const Zonotope &U, const Tensor &u, double time_step,
                     int taylor_terms) const;

    /// The first step of the algorithm "standard" or "wrapping_free" from Rinit with the input
    /// set U and the constant input u_trans (CORA's initReach); post continues from its state.
    ReachStep init_reach(const Zonotope &R_init, const std::optional<Zonotope> &U,
                         const std::optional<Tensor> &u_trans, const alg::LinearReach &alg) const;

    /// The step after state (CORA's post): the algorithm of alg, whose time_step and
    /// taylor_terms are those of init_reach.
    ReachStep post(const ReachState &state, const alg::LinearReach &alg) const;

    /// The error that the input solution picks up over a step, enclosed adaptively (CORA's
    /// errorSolution_adaptive): the Taylor series of Vdyn is cut where the gain falls below
    /// zeta_tabs, or at taylor_terms >= 2 if given.
    Zonotope error_solution_adaptive(const Zonotope &Vdyn, const std::optional<Zonotope> &Vstat,
                                     double time_step, double zeta_tabs,
                                     std::optional<int> taylor_terms = std::nullopt) const;

    /// The error of the trajectories of x' = A x + u from R0 against their linearization at t_c,
    /// for t in [t_min, t_max] (CORA's trajectoryError); f holds the flow at the start and the end
    /// (one entry: the same), the series stops at `order` >= 1 and n (dim, 1) is a direction.
    TrajectoryError trajectory_error(const std::vector<Tensor> &f, const Tensor &u,
                                     const Zonotope &R0, double t_min, double t_max, double t_c,
                                     int order, const Tensor &n) const;

    /// Equal if the matrices agree entrywise (A, B, C), up to tol (default: the repository
    /// tolerance); operator== and != use the default.
    bool isequal(const LinearSys &other, Tol tol = std::nullopt) const;
    bool operator==(const LinearSys &other) const { return isequal(other); }
    bool operator!=(const LinearSys &other) const { return !isequal(other); }

    /// A random system with B (n, m), C (p, n) and the eigenvalues of A in a box of the complex
    /// plane (CORA's generateRandom): n in 4..10, m in 1..3, p in 1..2 by default, the real parts
    /// in real_interval and the imaginary parts up to the smaller bound of imaginary_interval.
    static LinearSys
    generate_random(Rng &rng, std::optional<int64_t> state_dimension = std::nullopt,
                    std::optional<int64_t> input_dimension = std::nullopt,
                    std::optional<int64_t> output_dimension = std::nullopt,
                    const std::optional<Interval> &real_interval = std::nullopt,
                    const std::optional<Interval> &imaginary_interval = std::nullopt);

    /// The text that describes the system: its equations and matrices (CORA's display); also
    /// operator<< and, in Python, repr.
    std::string display() const;

    /// The system as a NonlinearSys, x' = A x (CORA's nonlinearSys); NonlinearSys has no inputs
    /// and no output map, so a system with B or a C other than the identity is refused with
    /// Error::not_supported. Not a batch of systems.
    NonlinearSys nonlinear_sys() const;

    /// Checks the arguments that every step operation shares: X in the state space, time_step > 0,
    /// taylor_terms >= 1; op names the operation in the message.
    void validate_step(const char *op, const Zonotope &X, double time_step, int taylor_terms) const;

    /// Checks a problem against the system, as reach does: the initial set, the input set and
    /// t_final. Algorithms registered from Python get the same checks.
    void validate(const ReachProblem &problem) const;

    /// Whether every specification holds for the outputs y = C x (+ B u reached from the initial
    /// set and the input set) up to t_final. `specs` are over the output space.
    VerifyResult verify(const ReachProblem &problem, const std::vector<Specification> &specs,
                        const alg::LinearVerify &alg) const;

  private:
    /// The dense A of a sparse system, formed once and shared by the copies of the system.
    struct DenseA {
        std::once_flag once;
        Tensor A;
    };
    Tensor A_; ///< undefined for a sparse system
    std::shared_ptr<const SparseMatrix> A_sparse_;
    std::shared_ptr<DenseA> dense_A_;
    std::optional<Tensor> B_, C_;
};

inline std::ostream &operator<<(std::ostream &out, const LinearSys &sys) {
    return out << sys.display();
}

/// canonical_form's result; defined after LinearSys, which it holds.
struct CanonicalForm {
    LinearSys sys;
    Zonotope U;
    Tensor u;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
