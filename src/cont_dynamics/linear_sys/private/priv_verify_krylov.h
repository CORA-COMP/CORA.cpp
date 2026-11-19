// priv_verify_krylov - the sound support bounds of the Krylov verification, step by step
//
// CORALean's LinearSys.verifyKrylovBoundsFloat for one direction w: in a restart window
// w' e^{tA} = nu V psi(t) + e, so a vector v of the initial set or of the input has the row
// W_v = nu v' V and the support term W_v psi plus e' v. On a step psi lies in the hull of its ends
// plus [F] psi_i; the ends are the iterates y_{i+1} = D y_i of the ball (D, r) of e^{H dt}, which
// carry the scaled error E_{i+1} = max(E_i, (||D|| + r) E_i + r ||y_i|| + gap). Each window starts
// from its own e_1 with no error. The input's integral gains the step's integrand times dt. Bit
// for bit at nearest on the lean backend.
//
// Syntax:   VerifyKrylovStepper s(vectors, windows, ratio, dt);   auto b = s.next();
// Inputs:   vectors - c, G, u_c and G_U of the initial set and the input (state space)
//           windows - the restart windows of w' e^{tA} on steps of ratio dt
//           ratio - steps of dt per step of the windows
//           dt - step size
// Outputs:  b - the bound of w' x(t) over the next step (empty: not computed, or no window left)
// See also: priv_verify_ra_krylov.cpp, priv_krylov_windows.cpp, priv_verify_exp_ball.h
//
// CORA.cpp only: the float definition of CORALean's LinearSys.verifyKrylovBoundsFloat.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The vectors and the windows ---------------------------------------------------------------------

/// The vectors of the initial set <c, G> and of the input <uc, GU> in the state space (columns),
/// with n_r = ||c|| + sum ||g_j|| and n_u = ||uc|| + sum ||g_U,j||, rounded up.
struct VerifyKrylovVectors {
    std::vector<double> c, uc;
    std::vector<std::vector<double>> G, GU;
    double n_r = 0, n_u = 0;
};

/// The vectors and their norms (LinearSys.verifyKrylovBoundsFloat's n_r and n_u).
VerifyKrylovVectors priv_verify_krylov_vectors(std::vector<double> c,
                                               std::vector<std::vector<double>> G,
                                               std::vector<double> uc,
                                               std::vector<std::vector<double>> GU);

// The bounds --------------------------------------------------------------------------------------

/// The bounds of one direction step by step (LinearSys.step_verifyKrylovStepsFloat).
class VerifyKrylovStepper {
  public:
    VerifyKrylovStepper(const VerifyKrylovVectors &vectors, const std::vector<KrylovWindow> &wins,
                        int64_t ratio, double dt);

    /// The bound over the next step; empty if no enclosure or no window is left.
    std::optional<double> next();
    /// Tuning: what no finer step removes of the last bound (the points, the error, the input's
    /// integral from below); not a certificate.
    double floor() const { return floor_; }

  private:
    /// A window's constants (step_verifyKrylovSetupFloat) and the rows of the vectors.
    bool start_window();

    const VerifyKrylovVectors &v_;
    const std::vector<KrylovWindow> &wins_;
    int64_t ratio_;
    double dt_;
    std::size_t window_ = 0;
    int64_t left_ = 0, k_ = 0;
    double integral_ = 0, integral_low_ = 0, floor_ = 0, E_ = 0, err_ = 0;
    std::vector<double> s_, D_, g_, y_, F_lo_, F_hi_;
    double r_d_ = 0, n_d_ = 0;
    std::vector<std::vector<double>> rows_;    ///< W_v of c, G, uc, GU in this order
    std::vector<double> mag_s_, mag_g_, at_y_; ///< sum |R| s, sum |R| g and W_v y_i per row
    std::vector<double> y2_, fy_lo_, fy_hi_, at_y2_, terms_; ///< the step's scratch
};

/// LinearSys.verifyKrylovBoundsFloat: the bounds of w' x(t) over N steps of dt (zeros for a zero
/// w), empty if a window's enclosures are not computed or the windows end before N steps.
std::optional<std::vector<double>> priv_verify_krylov_bounds(const VerifyKrylovVectors &vectors,
                                                             bool zero_direction,
                                                             const std::vector<KrylovWindow> &wins,
                                                             int64_t ratio, double dt, int64_t N);

// The trace ---------------------------------------------------------------------------------------

/// A direction as the oracle takes it: w, the plan of its windows (order, steps), the limit.
struct VerifyKrylovTraceDirection {
    std::vector<double> w;
    std::vector<std::pair<int64_t, int64_t>> plan;
    double limit = 0;
    bool strict = true;
    int64_t group = 0;
    std::vector<KrylovWindow> windows;         ///< the windows CORA.cpp computed
    std::optional<std::vector<double>> bounds; ///< the bounds of the last pass
};

/// What priv_verify_ra_krylov passed to its last pass, for the oracle tests.
struct VerifyKrylovTrace {
    std::vector<VerifyKrylovTraceDirection> dirs;
    int64_t groups = 0, halvings = 0, steps = 0;
    double dt0 = 0;
    std::vector<double> P, L; ///< the Lyapunov weight of A' if one was tried (empty if none)
    double a = 0;
    bool verified = false;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
