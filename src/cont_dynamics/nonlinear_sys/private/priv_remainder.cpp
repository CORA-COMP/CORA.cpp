// priv_remainder - the interval arithmetic of the Lagrange remainders of the linearization
//
// The helpers that priv_abstrerr_lin and priv_abstrerr_adaptive share: the entries of a box as
// (1, 1) ranges, the remainder of order 2 from the interval Hessian (1/2 dz' max|H_i| dz per
// component) and that of order 3 (1/6 sum_j dz_j (dz' T_ij dz)) from the interval third-order
// tensor. Products are evaluated from the left, as MATLAB multiplies.
//
// Syntax:   Tensor l = priv_second_order(H, dz, n);  auto V = priv_third_order(T, dz, n, false);
// Inputs:   H - the interval Hessian over a box (empty entries: identically 0)
//           T - the interval third-order tensor over a box
//           dz - the bounds of z - z* as (1, 1) ranges, or its largest absolute values as tensors
//           by_position - multiply by the dz of the position in ind{i}, as CORA's adaptive error
// Outputs:  l - the remainder of order 2 (n, 1)
//           V - the box of the remainder of order 3, none if T is all zero
// See also: priv_abstrerr_lin.cpp, priv_abstrerr_adaptive.cpp
//
// CORA: contDynamics/@contDynamics/private/priv_abstrerr_lin.m
// CORA: contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The boxes ---------------------------------------------------------------------------------------

// The entries of a column and back, the boxes of a set, and the ratio that decides a pass.
std::vector<Tensor> priv_entries(const Tensor &column) {
    const Tensor row = column.matrix_transpose();
    std::vector<Tensor> out;
    out.reserve(std::size_t(column.shape()[0]));
    for (int64_t i = 0; i < column.shape()[0]; ++i) out.push_back(row.slice(-1, i, i + 1));
    return out;
}

Tensor priv_column(const std::vector<Tensor> &entries) { return Tensor::concat(entries, -2); }

Tensor priv_abs_max(const Tensor &lo, const Tensor &hi) { return lo.abs().maximum(hi.abs()); }

std::vector<Range> priv_dz(const Interval &IHx, const std::optional<Interval> &IHu) {
    // the box of the state, then that of the input
    std::vector<Range> dz;
    for (const Interval *I : {&IHx, IHu ? &*IHu : nullptr}) {
        if (!I) continue;
        const std::vector<Tensor> lo = priv_entries(I->inf()), hi = priv_entries(I->sup());
        for (std::size_t j = 0; j < lo.size(); ++j) dz.emplace_back(lo[j], hi[j]);
    }
    return dz;
}

double priv_ratio_max(const Tensor &numerator, const Tensor &denominator) {
    // the largest ratio, skipping 0/0
    const std::vector<double> e = numerator.data(), a = denominator.data();
    double out = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < e.size(); ++i) {
        const double r = e[i] / a[i];
        if (!std::isnan(r) && (std::isnan(out) || r > out)) out = r;
    }
    return out;
}

// The orders --------------------------------------------------------------------------------------

Tensor priv_second_order(const std::vector<std::optional<Range>> &H, const std::vector<Tensor> &dz,
                         int64_t n) {
    // one remainder per component of f
    const auto d = int64_t(dz.size());
    std::vector<Tensor> l;
    l.reserve(std::size_t(n));
    for (int64_t i = 0; i < n; ++i) {
        // (dz' H) dz: the row dz' H first, as MATLAB multiplies from the left
        Tensor sum = dz[0].zeros_like();
        for (int64_t k = 0; k < d; ++k) {
            Tensor row = dz[0].zeros_like();
            for (int64_t j = 0; j < d; ++j)
                if (const auto &h = H[std::size_t((i * d + j) * d + k)])
                    row = row + dz[std::size_t(j)].multiply(priv_abs_max(h->lo, h->hi));
            sum = sum + row.multiply(dz[std::size_t(k)]);
        }
        l.push_back(sum * 0.5);
    }
    return priv_column(l);
}

std::optional<Interval> priv_third_order(const std::vector<std::optional<Range>> &T,
                                         const std::vector<Range> &dz, int64_t n,
                                         bool by_position) {
    const auto d = int64_t(dz.size());
    // the sum over the dimensions j with a nonzero tensor T_ij, per component
    const Range zero(dz[0].lo.zeros_like());
    std::vector<Tensor> lo, hi;
    bool any = false;
    for (int64_t i = 0; i < n; ++i) {
        Range sum = zero;
        int64_t position = 0; // the place of j in ind{i}
        for (int64_t j = 0; j < d; ++j) {
            const std::size_t base = std::size_t((i * d + j) * d * d);
            bool nonzero = false;
            for (int64_t kl = 0; kl < d * d && !nonzero; ++kl) nonzero = T[base + kl].has_value();
            if (!nonzero) continue;
            any = true;
            // (dz' T_ij) dz, the row first
            Range quad = zero;
            for (int64_t l = 0; l < d; ++l) {
                Range r = zero;
                for (int64_t k = 0; k < d; ++k)
                    if (const auto &t = T[base + std::size_t(k * d + l)]) r = r + dz[k] * *t;
                quad = quad + r * dz[std::size_t(l)];
            }
            sum = sum + quad * dz[std::size_t(by_position ? position : j)];
            ++position;
        }
        sum = sum * (1.0 / 6);
        lo.push_back(sum.lo);
        hi.push_back(sum.hi);
    }
    if (!any) return std::nullopt;
    return Interval(priv_column(lo), priv_column(hi));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
