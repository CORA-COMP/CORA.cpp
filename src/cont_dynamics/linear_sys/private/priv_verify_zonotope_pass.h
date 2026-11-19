// priv_verify_zonotope_pass - one sound pass of verify's zonotope algorithm on a given grid
//
// CORA's priv_verifyRA_zonotope computes the outer approximation by priv_reach_adaptive with a
// point expm and checks it against the specifications with linear programs and tolerances. Here
// CORALean's LinearSys.verifyZonotopePassFloat on the grid of CORA's tuning (the step sizes, the
// steps computed in full, the reduction budgets): e^{A dt} as a ball, e^{A t_k} as a product with
// a scaled error, the input's remainder kept; a full step's output set is checked by
// verifySpecHoldsFloat (a Farkas certificate for an unsafe set, never a tolerance). A violation
// is shown by LinearSys.verifyZonotopeWitnessFloat: one trajectory enclosed and checked. Bit for
// bit at nearest on the lean backend. The pass is empty if a step's enclosures are not computed.
//
// Syntax:   VerifyZonotopePass pass(sys);   auto steps = pass.run(specs, grid);
//           bool falsified = pass.witness(spec, t, b, beta);
// Inputs:   sys - A, x0, G0 (n, m), B (n, k), uc (k), GU (k, p), C (q, n), all row-major
//           specs - safe and unsafe sets a'y <= b of the outputs
//           grid - the steps: size, computed in full, reduction budget
// Outputs:  steps - per step its output set and flags, none if only propagated
//           falsified - whether the witness violates spec
// See also: priv_verify_ra_zonotope.cpp, priv_verify_exp_ball.h, priv_verify_correction.h
//
// CORA.cpp only: the float definitions of CORALean's LinearSys.verifyZonotopePassFloat,
// verifySpecHoldsFloat, verifyGirardReduceFloat and verifyZonotopeWitnessFloat.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cora/global/tensor/tensor.h"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The system, the specifications and the grid -----------------------------------------------------

/// x' = A x + B u, y = C x from <x0, G0>, u in <uc, GU>, on the host (row-major).
struct VerifyZonoSystem {
    Tensor A;                      ///< (n, n)
    std::vector<double> x0, G0;    ///< (n), (n, m)
    std::vector<double> B, uc, GU; ///< (n, k), (k), (k, p)
    std::vector<double> C;         ///< (q, n)
    int64_t n = 0, m = 0, k = 0, p = 0, q = 0;
};

/// The rows a'y <= b of a safe or an unsafe set of the outputs.
struct VerifySpecSet {
    bool safe = true;
    std::vector<std::vector<double>> a;
    std::vector<double> b;
};

/// A step of the grid: its size, whether its output set is computed and checked, the budget of
/// the reduction of the input's generators.
struct VerifyGridStep {
    double dt = 0;
    bool full = true;
    double budget = 0;
};

/// A zonotope <c, G> plus the box of radius r (LinearSys.VerifyZono).
struct VerifyZono {
    std::vector<double> c;
    std::vector<std::vector<double>> G;
    std::vector<double> r;
};

/// A full step's output set and which specifications hold on it.
struct VerifyZonoStep {
    VerifyZono Y;
    std::vector<char> flags;
};

// The pass ----------------------------------------------------------------------------------------

/// CORALean's LinearSys.verifySpecHoldsFloat: whether S is shown to hold on Z.
bool priv_verify_spec_holds(const VerifyZono &Z, const VerifySpecSet &S);

/// CORALean's LinearSys.verifyGirardReduceFloat: the generators P ++ new reduced into the box b
/// within the budget; returns the generators that stay, b holds the box.
std::vector<std::vector<double>> priv_verify_girard_reduce(std::vector<std::vector<double>> P,
                                                           std::vector<std::vector<double>> added,
                                                           std::vector<double> &b, double budget);

/// The passes of one verification: the scales and what each step size needs, computed once.
class VerifyZonotopePass {
  public:
    explicit VerifyZonotopePass(VerifyZonoSystem sys);

    /// LinearSys.verifyZonotopePassFloat on the grid; `failed` gets the step whose enclosures are
    /// not computed (the grid's size if the scales are not positive).
    std::optional<std::vector<std::optional<VerifyZonoStep>>>
    run(const std::vector<VerifySpecSet> &specs, const std::vector<VerifyGridStep> &grid,
        std::size_t *failed = nullptr);

    /// LinearSys.verifyZonotopeWitnessFloat: x0 + G0 b, u = uc + GU beta violates S at t.
    bool witness(const VerifySpecSet &S, double t, const std::vector<double> &b,
                 const std::vector<double> &beta);

    const VerifyZonoSystem &system() const { return sys_; }

    /// The Lyapunov weight P of A (row-major), its factor L of P - aI and a: checked by
    /// Krylov.lyapunovCertificateFloat; if accepted, the errors are also carried in it.
    void set_weight(std::vector<double> P, std::vector<double> L, double a);
    /// The weight given (P empty if none), for the oracle tests.
    const LyapunovWeight &weight() const { return weight_; }

  private:
    /// What a step of size dt needs: the ball of e^{A dt}, the input's solution, G u0, F.
    struct Step {
        std::optional<VerifyExpBall> ball;
        bool gu_ok = false;
        std::optional<VerifyInputSolution> input;
        std::vector<double> gu_lo, gu_hi;
        std::optional<VerifyCorrection> F;
        bool F_done = false;
    };
    Step &step(double dt, bool full);

    VerifyZonoSystem sys_;
    std::vector<double> s_, u0_, MU_; // the scales, [B uc] and [B GU] (n, p)
    bool scales_ok_ = true, has_u_ = false, has_m_ = false;
    VerifyPowers powers_;
    std::map<double, Step> steps_;
    LyapunovWeight weight_;
    std::optional<LyapunovCertificate> cert_;
};

// The trace ---------------------------------------------------------------------------------------

/// A witness tried: the specification, the time and the coefficients of the initial point and
/// the input.
struct VerifyZonoWitness {
    std::size_t spec = 0;
    double t = 0;
    std::vector<double> b, beta;
};

/// What priv_verify_ra_zonotope passed to the sound pass, for the oracle tests: the grids of its
/// passes, the witnesses it tried and the weight it gave (P empty if none).
struct VerifyZonotopeTrace {
    std::vector<std::vector<VerifyGridStep>> passes;
    std::vector<VerifyZonoWitness> witnesses;
    LyapunovWeight weight;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
