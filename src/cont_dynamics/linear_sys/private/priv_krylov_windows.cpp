// priv_krylov_windows - a Krylov trajectory in restart windows, each with a certified bound
//
// The bound of Krylov.errorBoundFloat grows with e^{t mu}, mu = max(0, mu_A, mu_T) the Gershgorin
// bounds of the quadratic forms of A and of the Hessenberg matrix T, so the trajectory is cut into
// windows as long as the bound at their end stays within the tolerance (tuning). Each window is
// then CORALean's LinearSys.step_verifyKrylovWindowFloat: the next start nu V D e_1 from the ball
// (D, r) of e^{H L}, plus the start's error |nu| r sum ||v_l|| and the width; the error carried by
// e^{L mu_A+} (the Gershgorin carry); where mu_A times the horizon exceeds 1 (RND02: 551, A
// stable) also in a certified Lyapunov weight (Krylov.windowErrorBoundFloat), the smaller of the
// two reported (step_verifyKrylovCarryFloat). Bit for bit at nearest (test_lean_linear_sys_verify_
// krylov).
//
// Syntax:   KrylovWindows traj(A, v, dt, steps, tol, krylov_step, 15, like);
//           while (!traj.done()) { KrylovWindow w = traj.next(); ... }
// Inputs:   A - the products with the matrix
//           v - the start vector (host)
//           time_step - the length of a step; a window is a whole number of them
//           steps - the number of steps of the horizon
//           tol - the tolerance of the bound of one window
//           krylov_step - how much the order grows per try
//           first_order - the order of the first try
//           like - backend, device and dtype of V and H
// Outputs:  window - V (n, k), H (k, k), |s|, the order, the steps it covers and the error bound
//           weighted() - whether the Lyapunov carry is in use; certificate() - its certificate
// See also: priv_krylov_subspace, priv_krylov_float
//
// CORA.cpp only: CORA has no restart; its single subspace runs to the final time.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The most steps m <= most whose bound at m dt meets the tolerance (the bound grows with m).
int64_t aux_valid_steps(const KrylovBoundFn &bound, double dt, int64_t most, double tol) {
    if (bound.ok(double(most) * dt, tol, kKrylovSquarings)) return most;
    int64_t low = 0, high = most;
    while (high - low > 1) {
        const int64_t mid = low + (high - low) / 2;
        (bound.ok(double(mid) * dt, tol, kKrylovSquarings) ? low : high) = mid;
    }
    return low;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The windows -------------------------------------------------------------------------------------

KrylovWindows::KrylovWindows(HostMatVec A, std::vector<double> v, double time_step, int64_t steps,
                             double tol, int krylov_step, int64_t first_order, const Tensor &like)
    : A_(std::move(A)), start_(std::move(v)), dt_(time_step), tol_(tol),
      mu_A_(std::max(0.0, A_.quad_bound())), steps_(steps), first_order_(first_order),
      step_(krylov_step), like_(like) {
    // the weight, if the Gershgorin carry would grow by more than e over the horizon
    if (mu_A_ * time_step * double(steps) > 1.0) {
        const LyapunovCertificate &c = A_.lyapunov();
        certificate_ = c;
        const RestartedBound candidate(c);
        // a weight that fails Krylov.windowErrorOkFloat (rate <= 0) is not accepted
        if (candidate.windows_ok()) weighted_.emplace(candidate);
    }
}

// The restart ------------------------------------------------------------------------------------

KrylovWindow KrylovWindows::next() {
    const int64_t remaining = steps_ - done_;
    const int64_t cap = std::min(kKrylovMaxOrder, A_.n);
    ArnoldiFloat state = priv_arnoldi_start(start_);
    // grow the order until the window ends the horizon, the cap is reached or it stops growing
    struct Try {
        int64_t k, m;
    };
    std::vector<Try> tried;
    int64_t k = std::min<int64_t>(std::max<int64_t>(first_order_, 1), cap);
    for (;; k = std::min<int64_t>(k + step_, cap)) {
        priv_arnoldi_extend(state, A_, k, machine_eps());
        const KrylovBoundFn bound(state, A_.quad_bound(), k);
        const int64_t before = tried.empty() ? 0 : tried.back().m;
        tried.push_back({k, aux_valid_steps(bound, dt_, remaining, tol_)});
        const int64_t m = tried.back().m;
        if (m >= remaining || k >= cap || (before >= 1 && double(m) < 1.1 * double(before))) break;
    }
    // the smallest order within 3 % of the longest window
    int64_t best = 0;
    for (const Try &t : tried) best = std::max(best, t.m);
    Try chosen = tried.back();
    for (const Try &t : tried) {
        if (double(t.m) < 0.97 * double(best)) continue;
        chosen = t;
        break;
    }
    k = chosen.k;
    const int64_t m = chosen.m;
    if (m < 1)
        CORA_FAIL(Error::not_converged,
                  "Krylov: not even one step of " + std::to_string(dt_) +
                      " meets the error bound krylov_error = " + std::to_string(tol_) +
                      " with order " + std::to_string(k) + " (growth rate " +
                      std::to_string(KrylovBoundFn(state, A_.quad_bound(), k).mu()) +
                      "); raise krylov_error or lower time_step");
    // the window: V (n, k) and H (k, k) of the first k vectors and columns
    const int64_t n = A_.n;
    std::vector<double> V(size_t(n * k)), H(size_t(k * k), 0.0);
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t r = 0; r < n; ++r) V[size_t(r * k + j)] = state.V[size_t(j)][size_t(r)];
        for (int64_t i = 0; i <= std::min(j + 1, k - 1); ++i)
            H[size_t(i * k + j)] = state.H[size_t(j)][size_t(i)];
    }
    KrylovWindow window{
        Tensor::like(like_, V, {n, k}), Tensor::like(like_, H, {k, k}), state.nu, k, m, 0.0, V, H};
    // LinearSys.step_verifyKrylovWindowFloat: the length m dt and its range checks
    double length = 0.0, length_down = 0.0;
    for (int64_t i = 0; i < m; ++i) length = dt_ + length;
    length_down = length;
    const double mu_raw = A_.quad_bound();
    const double mu = lean_maximum(0.0, lean_maximum(mu_raw, priv_hessenberg_quad_bound(state, k)));
    const double mu_plus = lean_maximum(mu_raw, 0.0);
    double range = 1.0;
    for (int i = 0; i < 29; ++i) range = range + range;
    const double squarings = double(int64_t(1) << (kKrylovSquarings - 1));
    sound_ = sound_ && length <= length_down && 0.0 <= dt_ && length * mu <= squarings &&
             length * mu_plus <= range;
    // the local error, the Gershgorin carry e^{L mu_A+} C + loc
    const double local = priv_krylov_error_bound(state, mu_raw, k, length, kKrylovSquarings);
    const double growth = std::exp(length * mu_plus);
    const double gershgorin = growth * error_ + local;
    // the next start nu V D e_1 from the ball (D, r) of e^{H L}, and the start's error
    const std::optional<VerifyExpBall> ball =
        priv_verify_exp_ball(window.H, std::vector<double>(size_t(k), 1.0), length);
    sound_ = sound_ && ball.has_value();
    double start_error = 0.0, gap = 0.0;
    std::vector<double> next(size_t(n), 0.0);
    if (ball) {
        const std::vector<double> D = ball->D.data();
        std::vector<double> width(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            const double d =
                lean_sum(k, [&](int64_t l) { return V[size_t(i * k + l)] * D[size_t(l * k)]; });
            next[size_t(i)] = state.nu * d;
            width[size_t(i)] = next[size_t(i)] - next[size_t(i)];
        }
        // |nu| r times the norms of V's columns, and the width of the start
        const double cols = lean_sum(k, [&](int64_t l) {
            std::vector<double> col(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) col[size_t(i)] = V[size_t(i * k + l)];
            return priv_krylov_norm_up(col);
        });
        start_error = (lean_absolute(state.nu) * ball->r) * cols;
        gap = priv_krylov_norm_up(width);
    }
    const double transition = local + (start_error + gap);
    window.error = gershgorin;
    if (weighted_) {
        // the weight's bound (step_verifyKrylovCarryFloat), the smaller of it and the Gershgorin
        const double inside = weighted_->window_bound(local);
        const double carried = RestartedBound::carry_min(gershgorin, inside);
        carries_.push_back({length, local, gershgorin, inside, carried, transition});
        window.error = carried;
        weighted_->step(length, transition);
    }
    // the Gershgorin error at the next start: of the trajectory, not of the min
    error_ = gershgorin + (start_error + gap);
    done_ += m;
    start_ = std::move(next);
    return window;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
