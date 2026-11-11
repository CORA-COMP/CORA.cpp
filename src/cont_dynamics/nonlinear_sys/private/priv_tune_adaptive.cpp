// priv_tune_adaptive - the formulas that tune the step size and orders of lin-adaptive
//
// The gain of the abstraction error over a change of the step size (CORA's aux_varphitotal), the
// gain that explains the error measured at a smaller step size (aux_varphiest, by bisection), the
// step size that minimizes the estimated size of the sets over the following steps
// (aux_optimaldeltat, eq. (49) of [1]) and the orders of the abstraction error (aux_getPowers).
// All of it is arithmetic on numbers, in the order CORALean's float definitions compute it.
//
// Syntax:   double dt = priv_optimal_delta_t(mu, red_factor, r_radius, rerr1, horizon, varphi,
// zeta_p); Inputs:   see the declarations in priv_adaptive.h Outputs:  see the declarations in
// priv_adaptive.h References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_lin_reach_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x^y for x > 0, as exp(y log x).
double aux_rpow(double x, double y) { return std::exp(y * std::log(x)); }

/// decr^i for any integer i.
double aux_dpow(double decr, int64_t i) {
    return i >= 0 ? priv_pow(decr, int(i)) : 1 / priv_pow(decr, int(-i));
}

/// The norm of the entries of v where the mask holds, the others 0.
double aux_masked_norm(const Vec &v, const std::vector<char> &mask) {
    Vec w(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) w[i] = mask[i] ? v[i] : 0.0;
    return priv_norm_up(w);
}

/// Whether the box [lo, hi] is not the point 0.
bool aux_nonzero(double lo, double hi) { return !(lo <= 0 && 0 <= lo && hi <= 0 && 0 <= hi); }

/// min(a, b) with -1 as Inf.
int aux_opt_min(int a, int b) { return a < 0 ? b : (b < 0 ? a : std::min(a, b)); }

} // namespace


// ===========================================  MAIN  =========================================== //

// The sums and powers -----------------------------------------------------------------------------

double priv_sum_up(const Vec &v) {
    double sum = 0;
    for (std::size_t i = v.size(); i-- > 0;) sum = v[i] + sum;
    return sum;
}

double priv_norm_up(const Vec &v) {
    Vec squares(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) squares[i] = std::abs(v[i]) * std::abs(v[i]);
    return std::sqrt(priv_sum_up(squares));
}

double priv_pow(double x, int k) {
    double power = 1;
    for (int i = 0; i < k; ++i) power = power * x;
    return power;
}

// The gain ----------------------------------------------------------------------------------------

double priv_varphi_total(double varphi, double varphi_lim, double prev, double curr, double decr) {
    const double scalings = std::log(prev / curr) / std::log(decr);
    const double eps10 = numerics::kScalingSlack;
    // the gain after i scalings, and the correction of the last one
    const auto interp = [&](int64_t i) {
        return varphi + (varphi_lim - varphi) * ((curr - aux_dpow(decr, i) * curr) / curr);
    };
    const auto correct = [&](double last, int64_t low, int64_t high) {
        return last + (1 - last) * ((prev - curr * aux_dpow(decr, high)) /
                                    (curr * (aux_dpow(decr, low) - aux_dpow(decr, high))));
    };
    Vec gains{varphi};
    const bool growing = !(scalings <= 0);
    // more scalings than one: the gains of 1 .. low, and the reciprocals of -1 .. low otherwise
    int64_t low;
    if (growing) {
        low = int64_t(std::floor(scalings + eps10));
        for (int64_t i = 1; i <= low; ++i) gains.push_back(interp(i));
        gains.back() = correct(gains.back(), low, low + 1);
    } else {
        low = int64_t(std::ceil(scalings - eps10));
        for (int64_t i = 1; i <= -low; ++i) gains.push_back(interp(-i));
        gains.back() = correct(gains.back(), low, low - 1);
    }
    double product = 1;
    for (const double g : gains) product = product * (growing ? g : 1 / g);
    return product;
}

double priv_varphi_est(double horizon, double deltat, const Vec &abstrerr_h, const Vec &abstrerr,
                       double decr, const std::vector<char> &mask, int minorder, int fuel) {
    const double rerr1 = aux_masked_norm(abstrerr_h, mask), rerrk = aux_masked_norm(abstrerr, mask);
    const double rhs = rerrk / rerr1, lim = priv_pow(decr, minorder + 1);
    // bisection on [0, lim]: the mid gain whose total is within the tolerances of rhs
    double low = 0, up = lim, prev = 0;
    for (int count = 0; count < fuel; ++count) {
        const double mid = low + 0.5 * (up - low);
        const double residual = priv_varphi_total(mid, lim, deltat, horizon, decr) - rhs;
        if (!(numerics::kGainResidual <= std::abs(residual)) ||
            (count > 0 && !(numerics::kGainStep <= std::abs(mid - prev))))
            return mid;
        if (!(0 <= residual))
            low = mid;
        else
            up = mid;
        prev = mid;
    }
    return low;
}

// The step size -----------------------------------------------------------------------------------

double priv_optimal_delta_t(double mu, double red_factor, double r_radius, double rerr1,
                            double deltat, double varphimin, double zeta_p) {
    const double zeta_z = 0.5 * red_factor;
    const int kprimemax = int(std::ceil(-std::log(100.0) / std::log(mu)));
    const double varphi_h = mu - varphimin, deltat0 = deltat * priv_pow(mu, 0);
    const auto kk = [&](int j) { return 1 / priv_pow(mu, j); };
    const auto deltats = [&](int j) { return deltat * priv_pow(mu, j); };
    const double a = 1 + (zeta_z + zeta_z);
    // the objective of the candidate j
    const auto objective = [&](int j) {
        const double k = kk(j);
        const int floork = int(std::floor(k));
        // the product of the gains over the first j + 1 scalings
        double varphiprod = 1;
        for (int i = 0; i <= j; ++i)
            varphiprod =
                varphiprod * ((varphimin + ((deltat0 - deltats(i)) / deltat0) * varphi_h) / mu);
        // all but the last of the scalings, as a sum from the first to the last
        double all_but_last = 0;
        for (int i = 0; i < floork; ++i) {
            const double ii = double(i + 1);
            all_but_last = all_but_last + aux_rpow(a, (k + 1) - ii) * aux_rpow(zeta_p, 1 - ii / k);
        }
        return (r_radius * aux_rpow(a, k)) * zeta_p +
               ((rerr1 / k) * varphiprod) *
                   (all_but_last + aux_rpow(1 + zeta_z, k - double(j)) * (k - double(floork)));
    };
    int best = 0;
    double best_value = objective(0);
    for (int j = 0; j <= kprimemax; ++j) {
        const double o = objective(j);
        if (!(best_value <= o)) {
            best = j;
            best_value = o;
        }
    }
    return deltats(best);
}

// The orders --------------------------------------------------------------------------------------

std::vector<int> priv_adaptive_orders(const Tensor &A, const std::vector<char> &zero_width,
                                      const std::vector<char> &gzero,
                                      const std::vector<std::optional<Range>> &H, int64_t n,
                                      int64_t m, int fuel) {
    const int64_t d = n + m;
    const Vec a = A.data();
    const auto entry = [&](int64_t i, int64_t j, int64_t k) {
        const auto &e = H[std::size_t((i * d + j) * d + k)];
        return std::pair<double, double>{e ? e->lo.item() : 0.0, e ? e->hi.item() : 0.0};
    };
    const bool all_zero =
        std::all_of(zero_width.begin(), zero_width.end(), [](char z) { return z; });
    const bool none_zero =
        std::none_of(zero_width.begin(), zero_width.end(), [](char z) { return z; });
    // sigma: the order of the error of the dynamics of each dimension
    std::vector<int> sigma(static_cast<std::size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        if (all_zero) {
            sigma[std::size_t(i)] = 2;
        } else if (none_zero) {
            sigma[std::size_t(i)] = 0;
        } else {
            // a nonzero diagonal Hessian entry of a dimension with width gives the order 0
            bool diagonal = false;
            for (int64_t j = 0; j < n; ++j) {
                const auto h = entry(i, j, j);
                diagonal =
                    diagonal || (!zero_width[std::size_t(j)] && aux_nonzero(h.first, h.second));
            }
            // else the least number of zero-width dimensions among the mixed entries
            int best = -1;
            for (int64_t j = 0; j < n && !diagonal; ++j)
                for (int64_t jj = j + 1; jj < n; ++jj) {
                    const auto p = entry(i, j, jj), q = entry(i, jj, j);
                    if (aux_nonzero(p.first + q.first, p.second + q.second)) {
                        const int count = zero_width[std::size_t(j)] + zero_width[std::size_t(jj)];
                        best = best < 0 ? count : std::min(best, count);
                    }
                }
            sigma[std::size_t(i)] = diagonal ? 0 : (best < 0 ? 2 : best);
        }
    }
    // the loop over the powers of A, as nonzero patterns
    std::vector<char> power(static_cast<std::size_t>(n * n), 0);
    for (int64_t i = 0; i < n; ++i) power[std::size_t(i * n + i)] = 1;
    // qi stays Inf (-1) until a power of A reaches a dimension with an error term
    std::vector<int> qi(static_cast<std::size_t>(n), -1);
    for (int p = 0; p < fuel; ++p) {
        bool done = true;
        for (int64_t i = 0; i < n; ++i) {
            int qip = -1;
            for (int64_t j = 0; j < n; ++j)
                if (!gzero[std::size_t(j)] && power[std::size_t(i * n + j)])
                    qip = aux_opt_min(qip, sigma[std::size_t(j)]);
            qi[std::size_t(i)] = aux_opt_min(qi[std::size_t(i)], qip < 0 ? -1 : qip + p);
            done = done && qi[std::size_t(i)] >= 0 && qi[std::size_t(i)] < p + 1;
        }
        if (done) return qi;
        // the next power: any k with power(i, k) and A(k, j) nonzero
        std::vector<char> next(static_cast<std::size_t>(n * n), 0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                for (int64_t k = 0; k < n; ++k)
                    if (power[std::size_t(i * n + k)] && a[std::size_t(k * n + j)] != 0)
                        next[std::size_t(i * n + j)] = 1;
        power = next;
    }
    return qi;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
