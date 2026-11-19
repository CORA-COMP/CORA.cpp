// linear_sys - lean::LinearSys, the reachability of x' = A x + B u computed by CORALean
//
// Syntax:     lean::LinearSys sys(A, B);   auto R = sys.reach(X0, t_final, alg::Standard{...});
//             lean::Zonotope Z = sys.homogeneous_solution(X, time_step, taylor_terms);
// Operations: reach (the homogeneous run), and one method per `linearSys.*` op of the oracle named
//             as CORA.cpp names CORA's: canonical_form, is_canonical_form, homogeneous_solution,
//             particular_solution_time_varying, particular_solution_constant, affine_solution,
//             one_step, one_step_closed_form, taylor_matrices, post, init_reach,
//             error_solution_adaptive, verify_correction, verify_zonotope_pass,
//             verify_zonotope_run
// Outputs:    lean zonotopes whose nominal part and error box together enclose the real result
// Scheme:     CORALean's sound float definitions; the error terms are tuned bounds of their own,
//             so they differ from CORA's (the nominal parts agree where no error term enters)
// See also:   lean/zonotope.h, contDynamics/linear_sys/linear_sys.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/global/oracle/request.h"
#include "cora/global/oracle/zonotope.h"

#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// The results -------------------------------------------------------------------------------------

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

/// verify_correction: the balancing scales of A and the interval matrices F, G of the support
/// function algorithm of verify with the orders their sums stopped at (empty where none).
struct VerifyCorrection {
    Tensor scales;
    std::optional<std::pair<int64_t, Interval>> F, G;
};

/// exp_ball: CORALean's MatExp.expBallFloat at the scales of MatExp.balanceScalesFloat, the
/// matrix D and the balanced radius r of e^{A dt} (empty where none).
struct VerifyExpBall {
    Tensor D;
    double r;
};

/// energy_growth: CORALean's MatExp.energyWeightFloat of A at the rate 1 / t_final (P, empty if A
/// has not the form) and MatExp.energyGrowthFloat's factor kappa at the balancing scales of A
/// (empty if the weight is not accepted).
struct VerifyEnergyGrowth {
    std::optional<Tensor> P;
    std::optional<double> kappa;
};

/// verify_support_func: the verdict of CORALean's LinearSys.verifySupportFuncFloat and CORA's
/// statistics; a falsification at the time point fals_step * time_step of halfspace fals_spec.
struct VerifySupportRun {
    bool verified = false, falsified = false;
    int64_t iterations = 0, num_steps = 0;
    double time_step = 0;
    std::optional<int64_t> fals_step, fals_spec;
};

/// A specification of the zonotope verification: the rows A y <= b (A (r, q), b (r, 1)) of a safe
/// or an unsafe set.
struct VerifyZonotopeSpec {
    bool safe = true;
    cora::Tensor A, b;
};

/// A step of a grid of the zonotope verification: its size, whether it is computed in full, the
/// budget of the reduction.
struct VerifyZonotopeGridStep {
    double time_step = 0;
    bool full = true;
    double budget = 0;
};

/// A full step of verify_zonotope_pass: the output set <c, G> plus the box r, and the flags.
struct VerifyZonotopeStep {
    Tensor c, r;
    std::vector<Tensor> G;
    std::vector<bool> flags;
};

/// A Lyapunov weight P of A, the factor L of P - a I and a (Krylov.lyapunovCertificateFloat).
struct VerifyWeight {
    cora::Tensor P, L;
    double a = 0;
};

/// A witness of verify_zonotope_run: the specification, the time, the coefficients b (m, 1) of the
/// initial point and beta (p, 1) of the constant input.
struct VerifyZonotopeWitness {
    int64_t spec = 0;
    double t = 0;
    cora::Tensor b, beta;
};

/// A direction of verify_krylov_pass: w (n, 1), the plan of its windows (order, steps), the limit
/// its bound must stay below (strictly, or at most), its group.
struct VerifyKrylovDirection {
    cora::Tensor w;
    std::vector<std::pair<int64_t, int64_t>> plan;
    double limit = 0;
    bool strict = true;
    int64_t group = 0;
};

/// What the oracle computed of a direction: its windows' order, nu and error (none if a window is
/// not computed) and its bounds (none if no enclosure).
struct VerifyKrylovDirectionResult {
    struct Window {
        int64_t order = 0;
        double nu = 0, err = 0;
    };
    std::optional<std::vector<Window>> windows;
    std::optional<std::vector<double>> bounds;
};

/// The verdict of verify_krylov_pass and every direction's windows and bounds.
struct VerifyKrylovPass {
    bool verified = false;
    std::vector<VerifyKrylovDirectionResult> dirs;
};

// The system --------------------------------------------------------------------------------------

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

    /// CORA's taylorMatrices at the order asked for, the remainder W from e^{|A| dt} by scaling
    /// and squaring (WCora, FCora, GCora; none without halvings).
    std::optional<TaylorMatrices> taylor_matrices_cora(double time_step, int taylor_terms) const;
    /// CORALean's MatExp.verifyCorrectionStateFloat and verifyCorrectionInputFloat at the scales
    /// of MatExp.balanceScalesFloat.
    VerifyCorrection verify_correction(double time_step) const;

    /// CORALean's MatExp.expBallFloat of A at dt (`linearSys.expBall`).
    std::optional<VerifyExpBall> exp_ball(double time_step) const;

    /// CORALean's energy weight of A and its growth factor (`linearSys.energyGrowth`).
    VerifyEnergyGrowth energy_growth(double t_final) const;

    /// CORALean's LinearSys.verifySupportFuncFloat of x' = A x + B u, u in U (no input without B
    /// or U), from <x0, G0> against the halfspaces (Cs C x)_q <= d_q over [0, t_final] (C = I
    /// without one), at most `fuel` passes.
    VerifySupportRun verify_support_func(const cora::Tensor &x0, const cora::Tensor &G0,
                                         const cora::Tensor &Cs, const cora::Tensor &d,
                                         double t_final, double min_step, int fuel,
                                         const std::optional<Zonotope> &U = std::nullopt) const;

    /// CORALean's LinearSys.verifyZonotopePassFloat of x' = A x + B u, y = C x from <x0, G0>, u in
    /// <uc, GU> (none without B) on the grid, errors also in an accepted weight: per step none
    /// (propagated only) or its output set and flags; none if a step's enclosures are not computed.
    std::optional<std::vector<std::optional<VerifyZonotopeStep>>>
    verify_zonotope_pass(const cora::Tensor &x0, const cora::Tensor &G0, const cora::Tensor &uc,
                         const cora::Tensor &GU, const std::vector<VerifyZonotopeSpec> &specs,
                         const std::vector<VerifyZonotopeGridStep> &grid,
                         const std::optional<VerifyWeight> &weight = std::nullopt) const;

    /// CORALean's LinearSys.verifyZonotopeRunFloat on the passes' grids and the witnesses: true
    /// (verified), false (a witness holds) or none (undecided).
    std::optional<bool>
    verify_zonotope_run(const cora::Tensor &x0, const cora::Tensor &G0, const cora::Tensor &uc,
                        const cora::Tensor &GU, const std::vector<VerifyZonotopeSpec> &specs,
                        const std::vector<std::vector<VerifyZonotopeGridStep>> &passes,
                        const std::vector<VerifyZonotopeWitness> &witnesses, double t_final,
                        const std::optional<VerifyWeight> &weight = std::nullopt) const;

    /// CORALean's LinearSys.verifyKrylovPassFloat of x' = A x + u, u in <uc, GU> (state space) from
    /// <x0, G0>: the directions' windows on steps of dt0, their bounds on `steps` steps of dt0 /
    /// 2^h.
    VerifyKrylovPass verify_krylov_pass(const cora::Tensor &x0, const cora::Tensor &G0,
                                        const cora::Tensor &uc, const cora::Tensor &GU,
                                        const std::vector<VerifyKrylovDirection> &dirs,
                                        int64_t groups, double dt0, int64_t halvings, int64_t steps,
                                        double eps, const std::optional<VerifyWeight> &cert) const;

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
