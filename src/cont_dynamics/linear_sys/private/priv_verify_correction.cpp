// priv_verify_correction - the interval matrices F and G of verify's support function algorithm
//
// Every rounded operation is the one of CORALean's definition at nearest, in its order: the sums
// over the inner index of a product from the last index down (CORALean's sumUp and sumDown nest
// to the right), the sums of the Taylor terms from the first term up.
//
// Syntax:   VerifyPowers P(A_rows, n);   auto F = priv_verify_correction_state(P, s, dt);
// Inputs:   A - row-major (n, n) system matrix
//           s - positive scales
//           dt - step size
// Outputs:  the order the sum stopped at and the bounds of the interval matrix, empty if none
// See also: priv_verify_correction.h, priv_verify_ra_support_function.cpp
//
// CORA.cpp only: the float definitions of CORALean's MatExp.verifyCorrection*Float.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/threads.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kFirstOrder = 3;  // the search of CORALean starts here (the remainder's order >= 2)
constexpr int kLastOrder = 74;  // CORA's limit: at order 75 a step counts as not converged
constexpr int kSweeps = 5;      // Osborne sweeps of balanceScalesFloat
constexpr int kFactorFuel = 64; // doublings or halvings of one visit

/// x A / d: each entry sums x_ik a_kj over k from n - 1 down to 0, zero terms skipped (adding a
/// zero leaves a sum that is not -0 as it is), then divides by d. Rows in parallel.
std::vector<double> aux_times_a(const std::vector<double> &x, const std::vector<int64_t> &row_ptr,
                                const std::vector<int64_t> &col_idx, const std::vector<double> &val,
                                int64_t n, double d) {
    std::vector<double> out(std::size_t(n) * n, 0.0);
    [[maybe_unused]] const int threads = threads_for(n, static_cast<long long>(n) * n);
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
    for (int64_t i = 0; i < n; ++i) {
        double *row = &out[std::size_t(i) * n];
        for (int64_t k = n - 1; k >= 0; --k) {
            const double xik = x[std::size_t(i) * n + k];
            if (xik == 0) continue;
            for (int64_t q = row_ptr[k]; q < row_ptr[k + 1]; ++q) row[col_idx[q]] += xik * val[q];
        }
        for (int64_t j = 0; j < n; ++j) row[j] /= d;
    }
    return out;
}

/// The power of two that brings row / f and col f within a factor of four (step_balanceFactor).
double aux_balance_factor(double row, double col) {
    double f = 1;
    for (int fuel = 0; fuel < kFactorFuel; ++fuel) {
        if (4.0 * col <= row) {
            row *= 0.5;
            col = col + col;
            f = f + f;
        } else if (4.0 * row <= col) {
            row = row + row;
            col *= 0.5;
            f *= 0.5;
        } else {
            break;
        }
    }
    return f;
}

// The Taylor sums ---------------------------------------------------------------------------------

/// The Taylor sums of F or G: term(eta) is the term of order eta; the sums stop at the first eta
/// whose term no longer changes them, and the remainder at order rem_order(eta), times scale,
/// widens them.
template <class Term>
std::optional<VerifyCorrection> aux_correction(VerifyPowers &P, const std::vector<double> &s,
                                               double dt, const Term &term, int rem_shift,
                                               double scale) {
    const int64_t n = P.n();
    const std::size_t nn = std::size_t(n) * n;
    const double eps = machine_eps(); // CORA's eps, 2^-52
    std::vector<double> lo(nn, 0.0), hi(nn, 0.0);
    for (int eta = 2; eta <= kLastOrder; ++eta) {
        // the term of order eta: [c a, 0] for every entry a of its power
        const auto [c, a] = term(eta);
        const std::vector<double> &pw = *a;
        bool converged = eta >= kFirstOrder;
        // CORA's stop: the term changes neither sum (its ends against eps times the sums)
        for (std::size_t e = 0; e < nn && converged; ++e) {
            const double p = c * pw[e], z = 0.0 * pw[e];
            converged = std::max(p, z) <= eps * hi[e] && eps * lo[e] <= std::min(p, z);
        }
        // stopped: the sums so far, widened by the remainder past their last order
        if (converged) {
            const std::optional<std::vector<double>> W =
                priv_verify_scaled_remainder(P.A(), s, n, dt, eta - rem_shift);
            if (!W) return std::nullopt;
            VerifyCorrection out{eta, std::vector<double>(nn), std::vector<double>(nn)};
            for (std::size_t e = 0; e < nn; ++e) {
                const double w = scale == 1.0 ? (*W)[e] : scale * (*W)[e];
                out.inf[e] = lo[e] - w;
                out.sup[e] = hi[e] + w;
            }
            return out;
        }
        // not yet: the term joins the sums, its negative part below and its positive part above
        for (std::size_t e = 0; e < nn; ++e) {
            const double p = c * pw[e];
            lo[e] = lo[e] + std::min(p, 0.0);
            hi[e] = hi[e] + std::max(p, 0.0);
        }
    }
    return std::nullopt;
}

/// The order at which a tight search stops as an untight one (tuning).
constexpr int kTightCap = 60;

/// LinearSys.step_verifyInputRemainderOkFloat: max_l dt sum_k W_lk (|u0_k| + sum_c |M_kc|) / s_l
/// at most eps max_l S_l / s_l, W the scaled remainder past order i - 1 (false if not admissible).
bool aux_remainder_ok(VerifyPowers &P, const std::vector<double> &s, double dt,
                      const std::vector<double> &u0, const std::vector<double> &M, int64_t k, int i,
                      const std::vector<double> &S, double eps) {
    const int64_t n = P.n();
    const std::optional<std::vector<double>> W =
        priv_verify_scaled_remainder(P.A(), s, n, dt, i - 1);
    if (!W) return false;
    std::vector<double> x(n);
    for (int64_t j = 0; j < n; ++j) {
        double rs = 0;
        for (int64_t c = k - 1; c >= 0; --c) rs = std::abs(M[std::size_t(j) * k + c]) + rs;
        x[j] = std::abs(u0[j]) + rs;
    }
    // maximumOver, nested to the right from the last row
    double r = 0, q = 0;
    for (int64_t l = n - 1; l >= 0; --l) {
        const double *Wl = &(*W)[std::size_t(l) * n];
        double sum = 0;
        for (int64_t j = n - 1; j >= 0; --j) sum = Wl[j] * x[j] + sum;
        const double rl = (dt * sum) / s[l], ql = S[l] / s[l];
        r = l == n - 1 ? rl : lean_maximum(rl, r);
        q = l == n - 1 ? ql : lean_maximum(ql, q);
    }
    return r <= eps * q;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The powers --------------------------------------------------------------------------------------

VerifyPowers::VerifyPowers(std::vector<double> A, int64_t n) : n_(n), A_(std::move(A)) {
    row_ptr_.assign(1, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            if (A_[std::size_t(i) * n + j] != 0) {
                col_idx_.push_back(j);
                val_.push_back(A_[std::size_t(i) * n + j]);
            }
        row_ptr_.push_back(static_cast<int64_t>(val_.size()));
    }
    powers_.push_back(A_);
}

const std::vector<double> &VerifyPowers::power(int i) {
    while (static_cast<int>(powers_.size()) < i) {
        const int next = static_cast<int>(powers_.size()) + 1;
        powers_.push_back(aux_times_a(powers_.back(), row_ptr_, col_idx_, val_, n_, next));
    }
    return powers_[i - 1];
}

// The scales and the remainder --------------------------------------------------------------------

std::vector<double> priv_verify_balance_scales(const std::vector<double> &A, int64_t n) {
    std::vector<double> s(n, 1.0);
    for (int sweep = 0; sweep < kSweeps; ++sweep)
        for (int64_t i = 0; i < n; ++i) {
            // the off-diagonal row and column sums of N = S^-1 |A| S, from the last index down
            double r = 0, c = 0;
            for (int64_t k = n - 1; k >= 0; --k) {
                if (k == i) continue;
                r = std::abs(A[std::size_t(i) * n + k]) * s[k] + r;
                c = std::abs(A[std::size_t(k) * n + i]) / s[k] + c;
            }
            const double row = r / s[i], col = c * s[i];
            if (row <= 0 || col <= 0) continue;
            s[i] = s[i] * aux_balance_factor(row, col);
        }
    return s;
}

std::optional<std::vector<double>> priv_verify_scaled_remainder(const std::vector<double> &A,
                                                                const std::vector<double> &s,
                                                                int64_t n, double dt, int eta) {
    // the norm of dt N, N_jk = |a_jk| s_k / s_j: the largest row sum, each from the last index down
    double nrm = 0;
    for (int64_t j = 0; j < n; ++j) {
        double sum = 0;
        for (int64_t k = n - 1; k >= 0; --k)
            sum = ((std::abs(A[std::size_t(j) * n + k]) * s[k]) / s[j]) * dt + sum;
        nrm = std::max(nrm, sum);
    }
    if (!(nrm + nrm <= double(eta + 2))) return std::nullopt;
    const double tau = priv_float_remainder_bound(nrm, eta);
    std::vector<double> W(std::size_t(n) * n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t k = 0; k < n; ++k) W[std::size_t(j) * n + k] = (s[j] / s[k]) * tau;
    return W;
}

// The correction matrices -------------------------------------------------------------------------

std::optional<VerifyCorrection>
priv_verify_correction_state(VerifyPowers &P, const std::vector<double> &s, double dt) {
    // order eta: [kappa dt^eta, 0] A^eta / eta!, the remainder at eta - 1
    return aux_correction(
        P, s, dt,
        [&](int eta) {
            const double c = priv_float_corr_coeff_low_tight(eta) * priv_float_power(eta, dt);
            return std::pair<double, const std::vector<double> *>(c, &P.power(eta));
        },
        1, 1.0);
}

std::optional<VerifyCorrection>
priv_verify_correction_input(VerifyPowers &P, const std::vector<double> &s, double dt) {
    // order eta: [kappa dt^eta / eta, 0] A^(eta-1) / (eta-1)!, dt times the remainder at eta - 2
    return aux_correction(
        P, s, dt,
        [&](int eta) {
            const double c =
                (priv_float_corr_coeff_low_tight(eta) * priv_float_power(eta, dt)) / double(eta);
            return std::pair<double, const std::vector<double> *>(c, &P.power(eta - 1));
        },
        2, dt);
}

// The input's solution ----------------------------------------------------------------------------

std::optional<VerifyInputSolution>
priv_verify_input_solution(VerifyPowers &P, const std::vector<double> &s, double dt,
                           const std::vector<double> &u0, const std::vector<double> &M, int64_t k,
                           bool tight) {
    const int64_t n = P.n();
    const std::size_t nk = std::size_t(n) * k;
    constexpr int kOrders = 75; // the search's fuel: orders 0 to 74
    const double eps = machine_eps();
    VerifyInputSolution b;
    b.Vs.assign(nk, 0.0);
    b.rho.assign(n, 0.0);
    b.w_lo.assign(n, 0.0);
    b.w_hi.assign(n, 0.0);
    // the nonzeros of u0 and of each column of M, from the last index down
    std::vector<int64_t> nzu;
    for (int64_t j = n - 1; j >= 0; --j)
        if (u0[j] != 0) nzu.push_back(j);
    std::vector<std::vector<int64_t>> nzm(k);
    for (int64_t j = n - 1; j >= 0; --j)
        for (int64_t c = 0; c < k; ++c)
            if (M[std::size_t(j) * k + c] != 0) nzm[c].push_back(j);
    std::vector<double> S(n, 0.0), wsum(n, 0.0);
    int eta = -1;
    for (int i = 0; i < kOrders && eta < 0; ++i) {
        // the coefficient dt^{i+1}/(i+1) and the power A^i/i! (the identity at i = 0)
        const double coef = priv_float_power(i + 1, dt) / double(i + 1);
        const std::vector<double> *Pi = i == 0 ? nullptr : &P.power(i);
        auto entry = [&](int64_t l, int64_t j) {
            return Pi ? coef * (*Pi)[std::size_t(l) * n + j] : (l == j ? coef * 1.0 : 0.0);
        };
        // [P_i] u0 and [P_i] M, each entry summed from the last index down
        std::vector<double> tw(n, 0.0), tv(nk, 0.0), a(n);
        for (int64_t l = 0; l < n; ++l) {
            double sum = 0;
            for (const int64_t j : nzu) sum = entry(l, j) * u0[j] + sum;
            tw[l] = sum;
            double rs = 0; // sumUp over the columns of |[P_i M]_lc|, from the last
            for (int64_t c = k - 1; c >= 0; --c) {
                double v = 0;
                for (const int64_t j : nzm[c]) v = entry(l, j) * M[std::size_t(j) * k + c] + v;
                tv[std::size_t(l) * k + c] = v;
                rs = std::abs(v) + rs;
            }
            a[l] = std::abs(tw[l]) + rs;
        }
        // CORA's stop: the order's row sums at most eps times those of the orders before it
        bool stop = i >= 1;
        for (int64_t l = 0; l < n && stop; ++l) stop = a[l] <= eps * S[l];
        // with tight (tuning): also the remainder past i - 1 at the level of eps, or i >= 60
        if (stop && tight && i < kTightCap) stop = aux_remainder_ok(P, s, dt, u0, M, k, i, S, eps);
        if (stop) {
            eta = i;
            break;
        }
        for (int64_t l = 0; l < n; ++l) {
            S[l] = S[l] + a[l];
            wsum[l] = wsum[l] + tw[l];
        }
        for (std::size_t e = 0; e < nk; ++e) b.Vs[e] = b.Vs[e] + tv[e];
        b.V.push_back(std::move(tv));
    }
    if (eta < 0) return std::nullopt;
    // the remainder past the orders i < eta: dt W |u0| and dt W sum_c |M_c|
    const std::optional<std::vector<double>> W =
        priv_verify_scaled_remainder(P.A(), s, n, dt, eta - 1);
    if (!W) return std::nullopt;
    std::vector<double> m_abs(n);
    for (int64_t j = 0; j < n; ++j) {
        double rs = 0;
        for (int64_t c = k - 1; c >= 0; --c) rs = std::abs(M[std::size_t(j) * k + c]) + rs;
        m_abs[j] = rs;
    }
    for (int64_t l = 0; l < n; ++l) {
        const double *Wl = &(*W)[std::size_t(l) * n];
        double ru = 0, rm = 0; // W |u0| and W sum_c |M_c|, from the last index down
        for (int64_t j = n - 1; j >= 0; --j) {
            ru = Wl[j] * std::abs(u0[j]) + ru;
            rm = Wl[j] * m_abs[j] + rm;
        }
        b.rho[l] = dt * rm;
        b.w_lo[l] = wsum[l] - dt * ru;
        b.w_hi[l] = wsum[l] + dt * ru;
    }
    return b;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
