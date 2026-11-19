// priv_verify_krylov - the sound support bounds of the Krylov verification, step by step
//
// Every rounded operation is the one of CORALean's definition at nearest, in its order: sums
// nested to the right (from the last term), maximumOver from the last element, CORALean's maximum,
// minimum and absolute, the corner products of Interval.mulLo and mulHi.
//
// Syntax:   VerifyKrylovStepper s(vectors, windows, ratio, dt);   auto b = s.next();
// Inputs:   vectors - the initial set and the input in the state space
//           windows - the restart windows of the direction
//           ratio - steps per step of the windows
//           dt - step size
// Outputs:  b - the bound over the next step
// See also: priv_verify_krylov.h, priv_verify_ra_krylov.cpp
//
// CORA.cpp only: the float definition of CORALean's LinearSys.verifyKrylovBoundsFloat.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cora/global/threads.h"

#include <algorithm>
#include <cmath>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vec = std::vector<double>;

/// Interval.mulLo: the least corner product of [a, b] and [c, d].
double aux_mul_lo(double a, double b, double c, double d) {
    return lean_minimum(lean_minimum(a * c, a * d), lean_minimum(b * c, b * d));
}

/// step_verifyKrylovMagFloat of a point row: sum_l |R_l| r_l.
double aux_mag(const Vec &R, const Vec &r) {
    return lean_sum(int64_t(R.size()), [&](int64_t l) {
        const double a = lean_absolute(R[l]);
        return lean_maximum(a, a) * r[l];
    });
}

/// step_verifyKrylovRowFloat: W_v = nu v' V, a point row at nearest. Each entry is summed from
/// the last index; the sums of all entries run together (the same order each), and a zero term is
/// skipped (a sum from +0 is never -0, so it changes no bit).
Vec aux_row(const KrylovWindow &W, const Vec &v) {
    const int64_t k = W.order, n = int64_t(v.size());
    Vec acc(static_cast<size_t>(k), 0.0);
    for (int64_t r = n - 1; r >= 0; --r) {
        const double x = v[size_t(r)];
        if (x == 0.0) continue;
        const double *Vr = &W.V_host[size_t(r * k)];
        for (int64_t l = 0; l < k; ++l) acc[size_t(l)] = x * Vr[l] + acc[size_t(l)];
    }
    for (int64_t l = 0; l < k; ++l)
        acc[size_t(l)] = aux_mul_lo(W.norm, W.norm, acc[size_t(l)], acc[size_t(l)]);
    return acc;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The vectors -------------------------------------------------------------------------------------

VerifyKrylovVectors priv_verify_krylov_vectors(Vec c, std::vector<Vec> G, Vec uc,
                                               std::vector<Vec> GU) {
    VerifyKrylovVectors out{std::move(c), std::move(uc), std::move(G), std::move(GU), 0, 0};
    auto norms = [](const Vec &x, const std::vector<Vec> &cols) {
        return priv_krylov_norm_up(x) + lean_sum(int64_t(cols.size()), [&](int64_t j) {
                   return priv_krylov_norm_up(cols[size_t(j)]);
               });
    };
    out.n_r = norms(out.c, out.G);
    out.n_u = norms(out.uc, out.GU);
    return out;
}

// The stepper -------------------------------------------------------------------------------------

VerifyKrylovStepper::VerifyKrylovStepper(const VerifyKrylovVectors &vectors,
                                         const std::vector<KrylovWindow> &wins, int64_t ratio,
                                         double dt)
    : v_(vectors), wins_(wins), ratio_(ratio), dt_(dt) {}

bool VerifyKrylovStepper::start_window() {
    const KrylovWindow &W = wins_[window_];
    const int64_t k = W.order;
    k_ = k;
    // the scales of H, positive, then the ball of e^{H dt} and [F]
    s_ = priv_verify_balance_scales(W.H_host, k);
    for (double x : s_)
        if (!(0.0 < x)) return false;
    const std::optional<VerifyExpBall> ball = priv_verify_exp_ball(W.H, s_, dt_);
    VerifyPowers powers(W.H_host, k);
    const std::optional<VerifyCorrection> F = priv_verify_correction_state(powers, s_, dt_);
    if (!ball || !F) return false;
    D_ = ball->D.data();
    r_d_ = ball->r;
    // ||S^-1 D S|| and g_l >= sum_m |F_lm| s_m
    n_d_ = lean_maximum(0.0, lean_max_over(k, [&](int64_t l) {
                            return lean_sum(k, [&](int64_t j) {
                                return (lean_absolute(D_[size_t(l * k + j)]) * s_[j]) / s_[l];
                            });
                        }));
    F_lo_ = F->inf;
    F_hi_ = F->sup;
    g_.assign(size_t(k), 0.0);
    for (int64_t l = 0; l < k; ++l)
        g_[size_t(l)] = lean_sum(k, [&](int64_t j) {
            const double a = lean_absolute(F_lo_[size_t(l * k + j)]);
            const double b = lean_absolute(F_hi_[size_t(l * k + j)]);
            return lean_maximum(a, b) * s_[j];
        });
    // the rows of the vectors [c, G, uc, GU], their magnitudes against s and g, and W_v e_1
    const size_t m = v_.G.size(), nr = m + 2 + v_.GU.size();
    rows_.assign(nr, Vec());
    [[maybe_unused]] const int threads =
        threads_for(int64_t(nr), static_cast<long long>(nr) * k * int64_t(v_.c.size()) / 256);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16) if (threads > 1)
    for (int64_t r = 0; r < int64_t(nr); ++r) {
        const auto i = size_t(r);
        const Vec &v = i == 0 ? v_.c : i <= m ? v_.G[i - 1] : i == m + 1 ? v_.uc : v_.GU[i - m - 2];
        rows_[i] = aux_row(W, v);
    }
    mag_s_.resize(nr);
    mag_g_.resize(nr);
    at_y_.resize(nr);
    for (size_t r = 0; r < nr; ++r) {
        mag_s_[r] = aux_mag(rows_[r], s_);
        mag_g_[r] = aux_mag(rows_[r], g_);
        // at y = e_1: R_0 1 + R_l 0 for the others, summed from the last
        at_y_[r] = lean_sum(k, [&](int64_t l) { return rows_[r][l] * (l == 0 ? 1.0 : 0.0); });
    }
    y_.assign(size_t(k), 0.0);
    y_[0] = 1.0;
    E_ = 0.0;
    err_ = W.error;
    left_ = W.steps * ratio_;
    ++window_;
    return true;
}

std::optional<double> VerifyKrylovStepper::next() {
    if (left_ == 0) {
        if (window_ >= wins_.size() || !start_window()) return std::nullopt;
    }
    const int64_t k = k_;
    // y_{i+1} = D y_i, its error E_{i+1}, and [F] y_i
    Vec &y2 = y2_;
    y2.resize(size_t(k));
    for (int64_t l = 0; l < k; ++l)
        y2[size_t(l)] = lean_sum(k, [&](int64_t j) { return D_[size_t(l * k + j)] * y_[j]; });
    const double ym =
        lean_max_over(k, [&](int64_t l) { return lean_absolute(y_[size_t(l)]) / s_[l]; });
    const double gap = lean_max_over(k, [&](int64_t l) { return (y2[l] - y2[l]) / s_[l]; });
    const double E2 = lean_maximum(E_, ((n_d_ + r_d_) * E_ + r_d_ * ym) + gap);
    // [F] y_i: the corners of [F_lj] y_j are F_lo y_j and F_hi y_j (a point y at nearest)
    Vec &fy_lo = fy_lo_, &fy_hi = fy_hi_;
    fy_lo.resize(size_t(k));
    fy_hi.resize(size_t(k));
    for (int64_t l = 0; l < k; ++l) {
        double lo = 0.0, hi = 0.0;
        for (int64_t j = k - 1; j >= 0; --j) {
            const double a = F_lo_[size_t(l * k + j)] * y_[j], b = F_hi_[size_t(l * k + j)] * y_[j];
            lo = lean_minimum(a, b) + lo;
            hi = lean_maximum(a, b) + hi;
        }
        fy_lo[size_t(l)] = lo;
        fy_hi[size_t(l)] = hi;
    }
    // per row its term on the step (step_verifyKrylovTermFloat; a point row R at nearest)
    const size_t nr = rows_.size(), m = v_.G.size();
    Vec &at_y2 = at_y2_, &terms = terms_;
    at_y2.resize(nr);
    terms.resize(nr);
    for (size_t r = 0; r < nr; ++r) {
        const Vec &R = rows_[r];
        // R y_{i+1} and the ends of R [F] y_i, each summed from the last index
        double c_y2 = 0.0, k_lo = 0.0, k_hi = 0.0;
        for (int64_t l = k - 1; l >= 0; --l) {
            const double x = R[size_t(l)];
            const double a = x * fy_lo[size_t(l)], b = x * fy_hi[size_t(l)];
            c_y2 = x * y2[size_t(l)] + c_y2;
            k_lo = lean_minimum(a, b) + k_lo;
            k_hi = lean_maximum(a, b) + k_hi;
        }
        at_y2[r] = c_y2;
        const double lo = lean_minimum(at_y_[r], at_y2[r]) + k_lo;
        const double hi = lean_maximum(at_y_[r], at_y2[r]) + k_hi;
        const double ext = mag_s_[r] * E2 + mag_g_[r] * E_;
        const bool signed_term = r == 0 || r == m + 1; // the centers c and uc
        terms[r] = (signed_term ? hi : lean_maximum(lean_absolute(lo), lean_absolute(hi))) + ext;
    }
    // the homogeneous bound and the integrand: c and G, then uc and GU
    const auto sum_terms = [&](size_t first, size_t count) {
        return terms[first] +
               lean_sum(int64_t(count), [&](int64_t j) { return terms[first + 1 + size_t(j)]; });
    };
    const double hom = err_ * v_.n_r + sum_terms(0, m);
    const double integrand = err_ * v_.n_u + sum_terms(m + 1, v_.GU.size());
    const double bound = (hom + integral_) + dt_ * lean_maximum(integrand, 0.0);
    // tuning: the points at the step's end, the error and the input's integral from below
    double points = err_ * v_.n_r + at_y2[0];
    for (size_t r = 1; r <= m; ++r) points += std::abs(at_y2[r]);
    const double step_low = at_y2[m + 1] + err_ * v_.n_u;
    floor_ = points + integral_low_ + dt_ * std::max(step_low, 0.0);
    integral_low_ += dt_ * step_low;
    std::swap(at_y_, at_y2);
    // on to the next step
    integral_ = integral_ + dt_ * integrand;
    std::swap(y_, y2);
    E_ = E2;
    --left_;
    return bound;
}

// The bounds --------------------------------------------------------------------------------------

std::optional<std::vector<double>> priv_verify_krylov_bounds(const VerifyKrylovVectors &vectors,
                                                             bool zero_direction,
                                                             const std::vector<KrylovWindow> &wins,
                                                             int64_t ratio, double dt, int64_t N) {
    if (!(0.0 < dt)) return std::nullopt;
    if (zero_direction) return std::vector<double>(size_t(N), 0.0);
    VerifyKrylovStepper stepper(vectors, wins, ratio, dt);
    std::vector<double> out;
    out.reserve(size_t(N));
    for (int64_t j = 0; j < N; ++j) {
        const std::optional<double> b = stepper.next();
        if (!b) return std::nullopt;
        out.push_back(*b);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
