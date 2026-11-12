// priv_krylov_windows - a Krylov trajectory in restart windows, each with a certified bound
//
// The bound of Krylov.errorBoundFloat grows with e^{t mu}, mu = max(0, mu_A, mu_T) the Gershgorin
// bounds of the quadratic forms of A and of the Hessenberg matrix T. On a long horizon mu_T alone
// makes it useless (the heat equation of HEAT03 has mu_A = 0 and mu_T near 2: the bound at 40 is
// 1e33 times the tolerance). The trajectory is cut into windows: Arnoldi restarts from the
// approximate state at the start of a window, which is as long as the bound at its end stays
// within the tolerance, and the error of that state is carried on by e^{t mu_A}
// (||e^{sA} e|| <= e^{s mu_A} ||e||), so that at every time of the window
//   ||e^{tA} v - |s| V e^{(t - t_w) T} e_1|| <= e^{(t - t_w) mu_A} E_w + bound_w(t - t_w)
// with E_w the error of the state at the start. For a nonexpansive A that is the sum of the local
// errors (CORALean's Krylov.restarted_error_le); CORALean has no float definition of the restart
// (oracle debt, docs/oracle-requests.md).
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
// See also: priv_krylov_subspace, priv_krylov_float
//
// CORA.cpp only: CORA has no restart; its single subspace runs to the final time.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
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
      step_(krylov_step), like_(like) {}

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
        Tensor::like(like_, V, {n, k}), Tensor::like(like_, H, {k, k}), state.nu, k, m, 0.0};
    // the error at every time of the window, and the state it ends in as the next start vector
    const double length = double(m) * dt_;
    const KrylovBoundFn bound(state, A_.quad_bound(), k);
    error_ = std::exp(length * mu_A_) * error_ + bound.at(length);
    window.error = error_;
    done_ += m;
    if (!done()) {
        const Tensor y = (window.H * length).expm().slice(-1, 0, 1);
        start_ = (window.V.matmul(y) * state.nu).data();
    }
    return window;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
