// linear_sys - lean::LinearSys, the reachability of x' = A x + B u computed by CORALean
//
// Syntax:     lean::LinearSys sys(A, B);   auto R = sys.reach(X0, t_final, alg::Standard{...});
//             lean::Zonotope Z = sys.homogeneous_solution(X, time_step, taylor_terms);
// Operations: reach (the homogeneous run), and one method per `linearSys.*` op of the oracle named
//             as CORA.cpp names CORA's: canonical_form, is_canonical_form, homogeneous_solution,
//             particular_solution_time_varying, particular_solution_constant, affine_solution,
//             one_step, one_step_closed_form, taylor_matrices, post, init_reach,
//             error_solution_adaptive
// Outputs:    lean zonotopes whose nominal part and error box together enclose the real result
// Scheme:     CORALean's sound float definitions; the error terms are tuned bounds of their own,
//             so they differ from CORA's (the nominal parts agree where no error term enters)
// See also:   lean/zonotope.h, contDynamics/linear_sys/linear_sys.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/global/oracle/request.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

struct ReachSet {
    std::vector<Zonotope> time_point, time_interval;
};

/// A step: the set at its end and over it.
struct Step {
    Zonotope time_point, time_interval;
};

/// canonical_form: the centered input set in the state space and the constant input as a box.
struct CanonicalForm {
    Zonotope U;
    Interval u;
};

/// taylor_matrices: the entrywise remainder W and the corrections F, G of CORA's taylorMatrices at
/// the admissible order eta_e (MatExp.expRemainderEntrywise, intervalCorrectionMatrixEntrywise,
/// intervalConstCorrectionMatrixEntrywise).
struct TaylorMatrices {
    int64_t order;
    Tensor W;
    Interval F, G;
};

class LinearSys {
  public:
    /// The dynamics matrix (n, n) in the current dtype.
    explicit LinearSys(const cora::Tensor &A);

    /// x' = A x + B u with y = C x; B (n, m) and C (p, n) are optional (no input; C = I).
    LinearSys(const cora::Tensor &A, const std::optional<cora::Tensor> &B,
              const std::optional<cora::Tensor> &C = std::nullopt);

    // Reach ---------------------------------------------------------------------------------------

    /// The reachable sets from X0 over [0, t_final] by the algorithm "standard" in steps of
    /// alg.time_step (t_final / time_step must be an integer); the Taylor series has
    /// alg.taylor_terms terms, zonotopes are reduced to alg.zonotope_order.
    ReachSet reach(const Zonotope &X0, double t_final, const alg::Standard &alg) const;

    /// As above for x' = A x + B u with u in U (a zonotope of the input space), by
    /// `linearSys.reach` (the system needs B): every trajectory under every signal in U.
    ReachSet reach(const Zonotope &X0, const Zonotope &U, double t_final,
                   const alg::Standard &alg) const;

    /// The input set B (U - c) and the constant input B (c + u_vec) (u_vec (m, 1)), as
    /// canonical_form.
    CanonicalForm canonical_form(const Zonotope &U, const cora::Tensor &u_vec) const;

    /// Whether B = I and C = I.
    bool is_canonical_form() const;

    /// The output set C R of a set of the state space (`linearSys.outputSet` with no input).
    Zonotope output_set(const cora::Zonotope &R) const;

    // Solutions and steps -------------------------------------------------------------------------

    /// [e^{A dt}] X.
    Zonotope homogeneous_solution(const Zonotope &X, double time_step, int taylor_terms) const;

    /// The sum of the terms [A^i dt^(i+1)/(i+1)!] V and the remainder; the oracle gives the same
    /// for the time-varying, the constant and the affine solution.
    Zonotope particular_solution_time_varying(const Zonotope &V, double time_step,
                                              int taylor_terms) const;
    Zonotope particular_solution_constant(const Zonotope &V, double time_step,
                                          int taylor_terms) const;
    Zonotope affine_solution(const Zonotope &V, double time_step, int taylor_terms) const;

    /// One step from X with the input set U and the constant input u (n, 1).
    Step one_step(const Zonotope &X, const Zonotope &U, const cora::Tensor &u, double time_step,
                  int taylor_terms) const;

    /// One step from X as `one_step`, in CORA's closed form (`linearSys.oneStepClosedForm`); none
    /// if no Taylor order within 4096 of taylor_terms encloses e^{|A| dt}.
    std::optional<Step> one_step_closed_form(const Zonotope &X, const Zonotope &U,
                                             const cora::Tensor &u, double time_step,
                                             int taylor_terms) const;

    /// CORA's F and G with the entrywise remainder (`linearSys.taylorMatrices`); none if no Taylor
    /// order within 4096 of taylor_terms encloses e^{|A| dt}.
    std::optional<TaylorMatrices> taylor_matrices(double time_step, int taylor_terms) const;

    /// [e^{A dt}] X at the end of the step, and the enclosure of the step.
    Step post(const Zonotope &X, double time_step, int taylor_terms) const;

    /// The window of step k from the first one, R_tau (none if no admissible order exists).
    std::optional<Zonotope> init_reach(const Zonotope &R_tau, double time_step, int taylor_terms,
                                       int step) const;

    /// The error set of the input solution, with Vstat if given.
    Zonotope error_solution_adaptive(const Zonotope &Vdyn, const std::optional<Zonotope> &Vstat,
                                     double time_step, int taylor_terms) const;

  private:
    /// The request `op` with the system (A and the matrices given) and the operands `more`.
    Reply ask_system(const char *op, std::initializer_list<Operand> more) const;

    Tensor A_;
    std::optional<Tensor> B_, C_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
