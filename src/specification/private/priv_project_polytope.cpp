// priv_project_polytope - the point of a polytope nearest to a point
//
// Syntax:   bool ok = priv_project_polytope(A, b, n, p, q);
// See also: priv_project_polytope.h
//
// CORA.cpp only: stands in for CORAquadprog with H = I and f = -p, as robustness asks it
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "specification/private/priv_project_polytope.h"

#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The least squares problem -----------------------------------------------------------------------

/// The least squares solution of B z = f for the columns cols of the (r, m) matrix E, by
/// Householder reflections; false if those columns are linearly dependent.
bool aux_least_squares(const std::vector<double> &E, std::size_t r, std::size_t m,
                       const std::vector<std::size_t> &cols, const std::vector<double> &f,
                       std::vector<double> &z) {
    const std::size_t k = cols.size();
    std::vector<double> B(r * k), g = f;
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < k; ++j) B[i * k + j] = E[i * m + cols[j]];
    double scale = 0;
    for (const double v : B) scale = std::max(scale, std::abs(v));
    // reflect column j to zero below its diagonal
    for (std::size_t j = 0; j < k; ++j) {
        double norm = 0;
        for (std::size_t i = j; i < r; ++i) norm += B[i * k + j] * B[i * k + j];
        norm = std::sqrt(norm);
        if (norm <= numerics::kRankDrop * scale) return false;
        const double alpha = B[j * k + j] > 0 ? -norm : norm;
        std::vector<double> v(r, 0.0);
        for (std::size_t i = j; i < r; ++i) v[i] = B[i * k + j];
        v[j] -= alpha;
        double vv = 0;
        for (std::size_t i = j; i < r; ++i) vv += v[i] * v[i];
        if (vv == 0) continue;
        // apply I - 2 v v'/(v'v) to the remaining columns and to g
        for (std::size_t c = j; c < k; ++c) {
            double s = 0;
            for (std::size_t i = j; i < r; ++i) s += v[i] * B[i * k + c];
            for (std::size_t i = j; i < r; ++i) B[i * k + c] -= 2 * s / vv * v[i];
        }
        double s = 0;
        for (std::size_t i = j; i < r; ++i) s += v[i] * g[i];
        for (std::size_t i = j; i < r; ++i) g[i] -= 2 * s / vv * v[i];
    }
    // back substitution in the triangle
    z.assign(k, 0.0);
    for (std::size_t j = k; j-- > 0;) {
        double s = g[j];
        for (std::size_t c = j + 1; c < k; ++c) s -= B[j * k + c] * z[c];
        z[j] = s / B[j * k + j];
    }
    return true;
}

// The nonnegative least squares problem -----------------------------------------------------------

/// Lawson and Hanson's NNLS: u >= 0 minimizing |E u - f| for the (r, m) matrix E.
std::vector<double> aux_nnls(const std::vector<double> &E, std::size_t r, std::size_t m,
                             const std::vector<double> &f) {
    std::vector<double> u(m, 0.0);
    std::vector<char> passive(m, 0), excluded(m, 0);
    const double eps = machine_eps();
    for (std::size_t outer = 0; outer < 10 * m + 10; ++outer) {
        // the gradient of the free columns
        std::vector<double> res = f;
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < m; ++j) res[i] -= E[i * m + j] * u[j];
        std::size_t best = m;
        double wbest = 1e3 * eps;
        for (std::size_t j = 0; j < m; ++j) {
            if (passive[j] || excluded[j]) continue;
            double w = 0;
            for (std::size_t i = 0; i < r; ++i) w += E[i * m + j] * res[i];
            if (w > wbest) wbest = w, best = j;
        }
        if (best == m) break;
        // the column joins the passive set, then the solution on that set is found
        passive[best] = 1;
        // solve on the passive columns, stepping back to the boundary while one turns negative
        bool dependent = false;
        while (true) {
            std::vector<std::size_t> cols;
            for (std::size_t j = 0; j < m; ++j)
                if (passive[j]) cols.push_back(j);
            // the passive columns are solved; dependent ones are put aside
            std::vector<double> z;
            if (!aux_least_squares(E, r, m, cols, f, z)) {
                passive[best] = 0, excluded[best] = 1, dependent = true;
                break;
            }
            // the step to the boundary of the non-negative orthant
            double alpha = 1;
            for (std::size_t c = 0; c < cols.size(); ++c)
                if (z[c] <= 0) alpha = std::min(alpha, u[cols[c]] / (u[cols[c]] - z[c]));
            for (std::size_t c = 0; c < cols.size(); ++c) u[cols[c]] += alpha * (z[c] - u[cols[c]]);
            if (alpha == 1) break;
            for (const std::size_t j : cols)
                if (u[j] <= 1e3 * eps) u[j] = 0, passive[j] = 0;
        }
        if (!dependent) std::fill(excluded.begin(), excluded.end(), 0);
    }
    return u;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool priv_project_polytope(const std::vector<double> &A, const std::vector<double> &b,
                           std::size_t n, const std::vector<double> &p, std::vector<double> &q) {
    const std::size_t m = b.size();
    q = p;
    // least distance problem of Lawson and Hanson: min |y| s.t. -A y >= A p - b, y = x - p
    std::vector<double> E((n + 1) * m), f(n + 1, 0.0);
    f[n] = 1;
    bool inside = true;
    for (std::size_t i = 0; i < m; ++i) {
        double h = -b[i];
        for (std::size_t j = 0; j < n; ++j) {
            E[j * m + i] = -A[i * n + j];
            h += A[i * n + j] * p[j];
        }
        E[n * m + i] = h;
        inside = inside && h <= 0;
    }
    if (inside) return true;
    const std::vector<double> u = aux_nnls(E, n + 1, m, f);
    // the residual of the NNLS problem gives the nearest point, none if it vanishes
    std::vector<double> res(n + 1);
    double norm = 0;
    for (std::size_t i = 0; i <= n; ++i) {
        res[i] = -f[i];
        for (std::size_t j = 0; j < m; ++j) res[i] += E[i * m + j] * u[j];
        norm += res[i] * res[i];
    }
    if (std::sqrt(norm) <= 1e3 * machine_eps() || res[n] == 0) return false;
    for (std::size_t j = 0; j < n; ++j) q[j] = p[j] - res[j] / res[n];
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
