// priv_distance - the Euclidean distance of a point to a polytope, by a least-distance program
//
// The point's shift z = y - x is the shortest vector with G z >= h, which Lawson and Hanson
// (Solving Least Squares Problems, ch. 23) reduce to nonnegative least squares.
//
// Syntax:   double d = priv_distance_to_polytope(H, x);
// Inputs:   H - a single nonempty polytope in halfspace form
//           x - a point
// Outputs:  d - min over y in H of |y - x|_2
// See also: Polytope::hausdorff_distance, priv_linear_programs
//
// CORA: contSet/@polytope/hausdorffDist.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A column-major matrix E (rows x cols) given by its columns.
using Columns = std::vector<Row>;

// Nonnegative least squares -----------------------------------------------------------------------

/// The least-squares coefficients of f on the columns of E selected by `passive`, by modified
/// Gram-Schmidt; a dependent column gets the coefficient 0.
Row aux_least_squares(const Columns &E, const std::vector<std::size_t> &passive, const Row &f) {
    const std::size_t k = passive.size();
    Columns q;
    Rows R(k, Row(k, 0.0));
    std::vector<bool> dependent(k, false);
    for (std::size_t j = 0; j < k; ++j) {
        Row v = E[passive[j]];
        const double original = std::sqrt(priv_dot(v, v));
        for (std::size_t i = 0; i < j; ++i) {
            if (dependent[i]) continue;
            R[i][j] = priv_dot(q[i], v);
            for (std::size_t t = 0; t < v.size(); ++t) v[t] -= R[i][j] * q[i][t];
        }
        const double norm = std::sqrt(priv_dot(v, v));
        if (norm <= numerics::kPolytopeRank * original || original == 0.0) {
            dependent[j] = true;
            q.emplace_back(v.size(), 0.0);
            continue;
        }
        R[j][j] = norm;
        // normalize the column, then keep it as a column of Q
        for (double &t : v) t /= norm;
        q.push_back(v);
    }
    // R s = Q' f, back substitution
    Row s(k, 0.0);
    for (std::size_t j = k; j-- > 0;) {
        if (dependent[j]) continue;
        double sum = priv_dot(q[j], f);
        for (std::size_t i = j + 1; i < k; ++i) sum -= R[j][i] * s[i];
        s[j] = sum / R[j][j];
    }
    return s;
}

/// min |E u - f| over u >= 0 (Lawson and Hanson's algorithm NNLS); E is given by its columns.
Row aux_nnls(const Columns &E, const Row &f) {
    const std::size_t m = E.size(), rows = f.size();
    Row u(m, 0.0);
    std::vector<bool> passive(m, false);
    // the gradient w = E'(f - E u)
    const auto gradient = [&](const Row &x) {
        Row residual = f;
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t t = 0; t < rows; ++t) residual[t] -= E[j][t] * x[j];
        Row w(m);
        for (std::size_t j = 0; j < m; ++j) w[j] = priv_dot(E[j], residual);
        return w;
    };
    double scale = 1.0;
    for (const Row &c : E) scale = std::max(scale, std::sqrt(priv_dot(c, c)));
    const double tol = numerics::kPolytopeNnls * scale * scale;
    // add the column of the largest gradient to the passive set until no gradient is positive
    for (std::size_t iteration = 0; iteration < 10 * (m + rows); ++iteration) {
        const Row w = gradient(u);
        std::size_t best = m;
        for (std::size_t j = 0; j < m; ++j)
            if (!passive[j] && w[j] > tol && (best == m || w[j] > w[best])) best = j;
        if (best == m) break;
        passive[best] = true;
        // solve on the passive columns and step back until the solution is nonnegative
        for (std::size_t guard = 0; guard < 10 * (m + 1); ++guard) {
            std::vector<std::size_t> index;
            for (std::size_t j = 0; j < m; ++j)
                if (passive[j]) index.push_back(j);
            const Row s = aux_least_squares(E, index, f);
            double alpha = 1.0;
            bool feasible = true;
            for (std::size_t i = 0; i < index.size(); ++i)
                if (s[i] <= 0.0) {
                    feasible = false;
                    alpha = std::min(alpha, u[index[i]] / (u[index[i]] - s[i]));
                }
            if (feasible) {
                for (std::size_t i = 0; i < index.size(); ++i) u[index[i]] = s[i];
                break;
            }
            // move toward s as far as u stays nonnegative, and drop what reached zero
            for (std::size_t i = 0; i < index.size(); ++i)
                u[index[i]] += alpha * (s[i] - u[index[i]]);
            for (const std::size_t j : index)
                if (u[j] <= numerics::kPolytopeNnls) {
                    u[j] = 0.0;
                    passive[j] = false;
                }
        }
    }
    return u;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The distance ------------------------------------------------------------------------------------

double priv_distance_to_polytope(const HostH &H, const Row &x) {
    // G z >= h for the shift z = y - x: -A z >= A x - b, and Ae z = be - Ae x both ways
    Rows G;
    Row h;
    for (std::size_t i = 0; i < H.A.size(); ++i) {
        Row g = H.A[i];
        for (double &a : g) a = -a;
        G.push_back(g);
        h.push_back(priv_dot(H.A[i], x) - H.b[i]);
    }
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        const double gap = H.be[i] - priv_dot(H.Ae[i], x);
        Row g = H.Ae[i], negated = H.Ae[i];
        for (double &a : negated) a = -a;
        G.push_back(g);
        h.push_back(gap);
        G.push_back(negated);
        h.push_back(-gap);
    }
    if (G.empty()) return 0.0;
    const std::size_t n = x.size(), m = G.size();
    // E = [G'; h'], f = (0, ..., 0, 1)
    Columns E(m, Row(n + 1));
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < n; ++i) E[j][i] = G[j][i];
        E[j][n] = h[j];
    }
    Row f(n + 1, 0.0);
    f[n] = 1.0;
    const Row u = aux_nnls(E, f);
    Row r = f;
    for (std::size_t j = 0; j < m; ++j)
        for (std::size_t t = 0; t <= n; ++t) r[t] -= E[j][t] * u[j];
    // r = f - E u; the shift is z = -(E u - f)[0..n) / (E u - f)[n] = -r[0..n) / r[n]
    const double last = -r[n];
    if (std::abs(last) <= numerics::kPolytopeRank)
        CORA_FAIL(Error::solver_issue, "Polytope: the least-distance program found no solution");
    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = r[i] / last;
        norm += z * z;
    }
    return std::sqrt(norm);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
