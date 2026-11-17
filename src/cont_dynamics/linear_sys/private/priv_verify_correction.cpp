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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
