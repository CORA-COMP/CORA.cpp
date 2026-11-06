// rescale - the same constrained zonotope with every factor scaled to the range it takes
//
// Syntax:   cZres = cz.rescale();   cZres = cz.rescale("iter");
// Inputs:   method - "exact" (default): the range of each factor is found by two linear programs;
//           "iter": a Gauss-Jordan elimination of [A b] with full pivoting (the generators follow
//           the columns), then one pass of interval arithmetic over the constraints
// Outputs:  cZres - the same set with factors in [-1, 1] that take every value of it: center
//           c + G m, generators G diag(r), constraints A diag(r) x = b - A m for the middle m and
//           radius r of the ranges. A set without constraints is Error::special_error and an
//           empty one Error::empty_set, as in CORA. CORA's cache of the ranges is not kept.
//           Not a batch
// See also: quad_map, compact
//
// CORA: contSet/@conZonotope/rescale.m
// CORA: contSet/@conZonotope/private/priv_ksi_optimizer.m
// CORA: contSet/@conZonotope/private/priv_ksi_iterative.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Elimination -------------------------------------------------------------------------------------

/// The constraints [A b] (p x (m+1), row-major) and the order of the generator columns.
struct Eliminated {
    std::vector<double> M;
    std::vector<int64_t> order;
};

/// Gauss-Jordan elimination of [A b] with the pivot of the largest row-normalized entry of the
/// remaining block (CORA's aux_rrefInfty); the columns are permuted with the pivots.
Eliminated aux_eliminate(const HostConZonotope &H, double eps) {
    const int64_t p = H.p, m = H.m, w = m + 1;
    Eliminated e;
    e.M.resize(std::size_t(p * w));
    double norm_inf = 0.0;
    for (int64_t i = 0; i < p; ++i) {
        double row = std::abs(H.b[i]);
        for (int64_t j = 0; j < m; ++j) {
            e.M[i * w + j] = H.A[i * m + j];
            row += std::abs(H.A[i * m + j]);
        }
        e.M[i * w + m] = H.b[i];
        norm_inf = std::max(norm_inf, row);
    }
    const double tol = double(std::max(p, w)) * eps * norm_inf;
    // the pivot is the entry of the largest share of its row in the block that is left
    e.order.resize(static_cast<std::size_t>(m));
    for (int64_t j = 0; j < m; ++j) e.order[j] = j;
    for (int64_t i = 0; i < p && i < m; ++i) {
        double best = 0.0;
        int64_t row_best = i, col_best = i;
        for (int64_t j = i; j < m; ++j) {
            for (int64_t r = i; r < p; ++r) {
                double sum = 0.0;
                for (int64_t k = i; k < m; ++k) sum += std::abs(e.M[r * w + k]);
                const double v = sum == 0.0 ? 0.0 : std::abs(e.M[r * w + j]) / sum;
                if (v > best) {
                    best = v;
                    row_best = r;
                    col_best = j;
                }
            }
        }
        if (best <= tol) continue;
        for (int64_t k = 0; k < w; ++k) std::swap(e.M[i * w + k], e.M[row_best * w + k]);
        for (int64_t r = 0; r < p; ++r) std::swap(e.M[r * w + i], e.M[r * w + col_best]);
        std::swap(e.order[i], e.order[col_best]);
        // the pivot row, scaled to 1, leaves every other row without this column
        const double pivot = e.M[i * w + i];
        std::vector<double> scaled(static_cast<std::size_t>(w));
        for (int64_t k = i; k < w; ++k) scaled[k] = e.M[i * w + k] / pivot;
        for (int64_t r = 0; r < p; ++r) {
            const double factor = e.M[r * w + i];
            for (int64_t k = i; k < w; ++k) e.M[r * w + k] -= factor * scaled[k];
        }
        for (int64_t k = i; k < w; ++k) e.M[i * w + k] = scaled[k];
    }
    return e;
}

/// The ranges of the factors by one pass of interval arithmetic over the constraints (CORA's
/// priv_ksi_iterative): factor j of row i lies in (b_i - sum_{k != j} a_ik [lo_k, hi_k]) / a_ij.
/// False if a range becomes empty.
bool aux_iterate(const std::vector<double> &M, int64_t p, int64_t m, std::vector<double> &lo,
                 std::vector<double> &hi) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const int64_t w = m + 1;
    // the ranges start unbounded; the factors start in the box
    std::vector<double> range_lo(std::size_t(m), -kInf), range_hi(std::size_t(m), kInf);
    lo.assign(std::size_t(m), -1.0);
    hi.assign(std::size_t(m), 1.0);
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            const double a = M[i * w + j];
            if (a == 0.0) continue;
            double sum_lo = 0.0, sum_hi = 0.0;
            for (int64_t k = 0; k < m; ++k) {
                if (k == j) continue;
                const double u = M[i * w + k] * lo[k], v = M[i * w + k] * hi[k];
                sum_lo += std::min(u, v);
                sum_hi += std::max(u, v);
            }
            const double u = (M[i * w + m] - sum_hi) / a, v = (M[i * w + m] - sum_lo) / a;
            range_lo[j] = std::max(range_lo[j], std::min(u, v));
            range_hi[j] = std::min(range_hi[j], std::max(u, v));
            lo[j] = std::max(lo[j], range_lo[j]);
            hi[j] = std::min(hi[j], range_hi[j]);
            if (lo[j] > hi[j]) return false;
        }
    }
    return true;
}

// Scaling -----------------------------------------------------------------------------------------

/// A row (1, m) or column (m, 1) of host values like `like`.
Tensor aux_vector(const Tensor &like, const std::vector<double> &v, bool column) {
    const int64_t m = int64_t(v.size());
    return Tensor::like(like, v, column ? Shape{m, 1} : Shape{1, m});
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::rescale(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(method == "exact" || method == "iter", Error::wrong_value, [&] {
        return "rescale: the method must be 'exact' or 'iter', got '" + method + "'";
    });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("rescale: a linear program per factor; it supports a single set, not a "
                           "batch");
    });
    const HostConZonotope H = priv_host_con_zonotope(*this);
    CORA_CHECK(!priv_is_empty(H), Error::empty_set,
               [] { return std::string("rescale: the set is empty"); });
    CORA_CHECK(A_.numel() != 0, Error::special_error,
               [] { return std::string("rescale: no constraints left!"); });

    // the ranges of the factors, of a set whose generators and constraints may be permuted
    Tensor G = G_, A = A_, b = b_;
    std::vector<double> lo, hi;
    if (method == "exact") {
        std::tie(lo, hi) = priv_factor_ranges(H);
    } else {
        const Eliminated e = aux_eliminate(H, machine_eps(c_.dtype()));
        const int64_t p = H.p, m = H.m, w = m + 1;
        std::vector<double> a(std::size_t(p * m)), rhs(static_cast<std::size_t>(p));
        for (int64_t i = 0; i < p; ++i) {
            for (int64_t j = 0; j < m; ++j) a[i * m + j] = e.M[i * w + j];
            rhs[i] = e.M[i * w + m];
        }
        G = G_.take(e.order, -1);
        A = Tensor::like(A_, a, {p, m});
        b = Tensor::like(b_, rhs, {p, 1});
        if (!aux_iterate(e.M, p, m, lo, hi))
            CORA_FAIL(Error::empty_set, "rescale: the set is empty");
    }
    std::vector<double> middle(lo.size()), radius(lo.size());
    for (std::size_t j = 0; j < lo.size(); ++j) {
        middle[j] = (hi[j] + lo[j]) / 2;
        radius[j] = (hi[j] - lo[j]) / 2;
    }
    // the new factors are (x - middle) / radius: shift the center and offset, scale the columns
    const Tensor m_col = aux_vector(c_, middle, true), r_row = aux_vector(c_, radius, false);
    return {c_ + G.matmul(m_col), G.multiply(r_row), A.multiply(r_row), b - A.matmul(m_col)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
