// priv_vertices - vertices from halfspaces and halfspaces from vertices, on the host
//
// Syntax:   Row v = priv_vertices_1d(H, empty);   Rows V = priv_vertices(H);
//           HostH H = priv_constraints_of_points(like, points, n);
// Inputs:   H - a single polytope in halfspace form (bounded for priv_vertices)
//           points - rows of n numbers
//           like - a tensor of the backend that does the singular value decomposition
// Outputs:  V - the vertices, or the halfspaces of their convex hull
// See also: priv_compact, Polytope::constraints
//
// CORA: contSet/@polytope/private/priv_vertices_1D.m
// CORA: contSet/@polytope/vertices_.m
// CORA: contSet/@polytope/constraints.m
// CORA authors: Mark Wetzlinger, Matthias Althoff, Niklas Kochdumper, Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMaxCombinations = 2e6; // subsets of rows or points tried one by one

// Small helpers -----------------------------------------------------------------------------------

/// The number of ways to pick k of m (as a double, for the size check).
double aux_choose(int64_t m, int64_t k) {
    double count = 1.0;
    for (int64_t i = 1; i <= k; ++i) count = count * double(m - k + i) / double(i);
    return count;
}

/// The unit vector orthogonal to the r - 1 rows of M (each of r numbers), none if they are
/// dependent (reduced row echelon form with full pivoting).
std::optional<Row> aux_null_vector(Rows M, std::size_t r) {
    const std::size_t rows = M.size();
    std::vector<std::size_t> pivot_column;
    double scale = 0.0;
    for (const Row &row : M)
        for (const double x : row) scale = std::max(scale, std::abs(x));
    std::vector<bool> used(r, false);
    // reduced row echelon form: the largest entry of the rest is the next pivot
    for (std::size_t i = 0; i < rows; ++i) {
        std::size_t best_row = i, best_col = r;
        double best = 0.0;
        for (std::size_t a = i; a < rows; ++a)
            for (std::size_t c = 0; c < r; ++c)
                if (!used[c] && std::abs(M[a][c]) > best) {
                    best = std::abs(M[a][c]);
                    best_row = a;
                    best_col = c;
                }
        if (best <= numerics::kPolytopeRank * scale) return std::nullopt;
        std::swap(M[i], M[best_row]);
        used[best_col] = true;
        pivot_column.push_back(best_col);
        // clear the pivot's column in the other rows
        for (std::size_t a = 0; a < rows; ++a) {
            if (a == i) continue;
            const double f = M[a][best_col] / M[i][best_col];
            for (std::size_t c = 0; c < r; ++c) M[a][c] -= f * M[i][c];
        }
    }
    // the free column is 1, each pivot column solves its row
    std::size_t free_column = 0;
    while (free_column < r && used[free_column]) ++free_column;
    Row v(r, 0.0);
    v[free_column] = 1.0;
    for (std::size_t i = 0; i < rows; ++i)
        v[pivot_column[i]] = -M[i][free_column] / M[i][pivot_column[i]];
    const double norm = priv_norm(v);
    for (double &x : v) x /= norm;
    return v;
}

/// The rows of Ae that are linearly independent (the first ones that are), by Gram-Schmidt.
std::vector<std::size_t> aux_independent_rows(const Rows &rows) {
    std::vector<std::size_t> kept;
    Rows basis;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        Row v = rows[i];
        const double norm = priv_norm(v);
        for (const Row &b : basis) {
            const double f = priv_dot(v, b);
            for (std::size_t k = 0; k < v.size(); ++k) v[k] -= f * b[k];
        }
        const double rest = priv_norm(v);
        if (norm > 0.0 && rest > numerics::kPolytopeRank * norm) {
            for (double &x : v) x /= rest;
            basis.push_back(v);
            kept.push_back(i);
        }
    }
    return kept;
}

/// Whether the point x satisfies every row of H, up to the slack of the vertex test.
bool aux_feasible(const HostH &H, const Row &x) {
    for (std::size_t i = 0; i < H.A.size(); ++i)
        if (priv_dot(H.A[i], x) >
            H.b[i] + numerics::kPolytopeVertexInside * (1.0 + std::abs(H.b[i])))
            return false;
    for (std::size_t i = 0; i < H.Ae.size(); ++i)
        if (std::abs(priv_dot(H.Ae[i], x) - H.be[i]) >
            numerics::kPolytopeVertexInside * (1.0 + std::abs(H.be[i])))
            return false;
    return true;
}

/// Whether x is already in the list of vertices.
bool aux_listed(const Rows &vertices, const Row &x) {
    for (const Row &v : vertices) {
        double distance = 0.0, scale = 1.0;
        for (std::size_t k = 0; k < x.size(); ++k) {
            distance = std::max(distance, std::abs(v[k] - x[k]));
            scale = std::max(scale, std::abs(x[k]));
        }
        if (distance <= numerics::kPolytopeVertexMerge * scale) return true;
    }
    return false;
}

// Halfspaces of points ----------------------------------------------------------------------------

/// The halfspaces of a one-dimensional hull (CORA's aux_1D): the points may be infinite.
HostH aux_constraints_1d(const Rows &points) {
    // the least and the greatest vertex
    HostH H;
    H.n = 1;
    double lo = kInf, hi = -kInf;
    for (const Row &p : points) {
        lo = std::min(lo, p[0]);
        hi = std::max(hi, p[0]);
    }
    // R, a half line, a point or a segment
    const bool hi_inf = hi == kInf, lo_inf = lo == -kInf;
    if (hi_inf && lo_inf) return H;
    if (hi_inf) {
        H.A = {{-1.0}};
        H.b = {-lo};
    } else if (lo_inf) {
        H.A = {{1.0}};
        H.b = {hi};
    } else if (within_tol(hi, lo, machine_eps())) {
        H.Ae = {{1.0}};
        H.be = {hi};
    } else {
        H.A = {{1.0}, {-1.0}};
        H.b = {hi, -lo};
    }
    return H;
}

/// The halfspaces of a hull in the plane that is a point or a segment (CORA's
/// aux_2D_degenerate): a box around a point, or the segment's line and its two caps.
HostH aux_constraints_2d_degenerate(const Rows &points) {
    // all points the same: four rows around it
    HostH H;
    H.n = 2;
    const Row &first = points[0];
    bool same = true;
    for (const Row &p : points)
        same = same && within_tol(p[0], first[0], machine_eps()) &&
               within_tol(p[1], first[1], machine_eps());
    if (same) {
        H.A = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        H.b = {first[0], first[1], -first[0], -first[1]};
        return H;
    }
    // the ends along x, or along y if x is constant
    const auto extreme = [&](std::size_t coordinate, bool largest) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < points.size(); ++k)
            if (largest ? points[k][coordinate] > points[best][coordinate]
                        : points[k][coordinate] < points[best][coordinate])
                best = k;
        return best;
    };
    std::size_t lo = extreme(0, false), hi = extreme(0, true);
    if (lo == hi) {
        lo = extreme(1, false);
        hi = extreme(1, true);
    }
    Row direction = {points[hi][0] - points[lo][0], points[hi][1] - points[lo][1]};
    const double norm = priv_norm(direction);
    direction = {direction[0] / norm, direction[1] / norm};
    const Row side = {-direction[1], direction[0]};
    H.A = {direction, {-direction[0], -direction[1]}, side, {-side[0], -side[1]}};
    H.b = {priv_dot(direction, points[hi]), -priv_dot(direction, points[lo]),
           priv_dot(side, points[hi]), -priv_dot(side, points[hi])};
    // the four sides of the rectangle around the points
    return H;
}

/// The facets a.y <= d of the hull of the points y (rows of r numbers, full-dimensional in R^r),
/// every facet once, by the hyperplanes through r of the points that have all points on one side.
std::vector<std::pair<Row, double>> aux_facets(const Rows &y, std::size_t r) {
    std::vector<std::pair<Row, double>> facets;
    const int64_t p = int64_t(y.size());
    if (r == 1) {
        double lo = kInf, hi = -kInf;
        for (const Row &q : y) {
            lo = std::min(lo, q[0]);
            hi = std::max(hi, q[0]);
        }
        return {{{1.0}, hi}, {{-1.0}, -lo}};
    }
    if (aux_choose(p, int64_t(r)) > kMaxCombinations)
        CORA_FAIL(Error::not_supported,
                  "Polytope: too many vertices for the halfspaces of their hull by combinations");
    double scale = 1.0;
    for (const Row &q : y)
        for (const double x : q) scale = std::max(scale, std::abs(x));
    // the hyperplane through r points: the normal is orthogonal to their differences
    Combinations subsets(p, int64_t(r));
    for (std::vector<int64_t> flat = subsets.next(4096); !flat.empty(); flat = subsets.next(4096)) {
        for (std::size_t s = 0; s < flat.size(); s += r) {
            Rows differences;
            for (std::size_t k = 1; k < r; ++k) {
                Row d(r);
                for (std::size_t j = 0; j < r; ++j) d[j] = y[flat[s + k]][j] - y[flat[s]][j];
                differences.push_back(d);
            }
            const std::optional<Row> normal = aux_null_vector(differences, r);
            if (!normal) continue;
            double offset = priv_dot(*normal, y[flat[s]]);
            double high = -kInf, low = kInf;
            for (const Row &q : y) {
                high = std::max(high, priv_dot(*normal, q) - offset);
                low = std::min(low, priv_dot(*normal, q) - offset);
            }
            const double slack = numerics::kPolytopeFacet * scale;
            Row a = *normal;
            if (high <= slack) {
                // all points below: a.y <= offset
            } else if (low >= -slack) {
                for (double &x : a) x = -x;
                offset = -offset;
            } else {
                continue;
            }
            bool known = false;
            for (const auto &[b, d] : facets) {
                double distance = std::abs(d - offset);
                for (std::size_t j = 0; j < r; ++j)
                    distance = std::max(distance, std::abs(b[j] - a[j]));
                known = known || distance <= numerics::kPolytopeFacet * scale;
            }
            if (!known) facets.emplace_back(a, offset);
        }
    }
    return facets;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// One dimension -----------------------------------------------------------------------------------

// Vertices of one dimension -----------------------------------------------------------------------

Row priv_vertices_1d(const HostH &H, bool &empty) {
    HostH M = H;
    empty = priv_compact_1d(M, numerics::kPolytopeAligned);
    if (empty) return {};
    priv_normalize_rows(M);
    const double tol = numerics::kPolytopeAligned;
    Row v;
    if (!M.A.empty()) {
        // bounded from below by -x <= b, else unbounded
        double lower = -kInf;
        for (std::size_t i = 0; i < M.A.size(); ++i)
            if (within_tol(M.A[i][0], -1.0, tol)) lower = -M.b[i];
        v.push_back(lower);
        bool bounded_above = false;
        for (std::size_t i = 0; i < M.A.size(); ++i)
            if (within_tol(M.A[i][0], 1.0, tol)) {
                bounded_above = true;
                if (!within_tol(lower, M.b[i], machine_eps())) v.push_back(M.b[i]);
            }
        if (!bounded_above) v.push_back(kInf);
    } else if (!M.Ae.empty()) {
        v.push_back(M.be[0] / M.Ae[0][0]);
    } else {
        CORA_FAIL(Error::special_error, "Error in vertex computation of 1D polytope.");
    }
    return v;
}

// Several dimensions ------------------------------------------------------------------------------

Rows priv_vertices(const HostH &H) {
    if (priv_is_empty(H)) return {};
    if (!priv_is_bounded(H))
        CORA_FAIL(Error::not_supported, "Vertex enumeration requires a bounded polytope.");
    // few rows: the minimal representation
    HostH M = H;
    priv_normalize_rows(M);
    bool minimal = false;
    if (priv_compact_all(M, numerics::kPolytopeAligned, minimal)) return {};
    const std::vector<std::size_t> independent = aux_independent_rows(M.Ae);
    const int64_t k = H.n - int64_t(independent.size());
    const int64_t count = int64_t(M.A.size());
    if (k < 0 || k > count)
        CORA_FAIL(Error::not_supported, "Polytope: no vertex enumeration for this polytope");
    if (aux_choose(count, k) > kMaxCombinations)
        CORA_FAIL(Error::not_supported,
                  "Polytope: too many rows for the vertex enumeration by combinations");
    Rows vertices;
    Combinations subsets(count, k);
    // k = 0 is one empty subset: the equalities alone
    std::vector<int64_t> flat = k == 0 ? std::vector<int64_t>{} : subsets.next(4096);
    do {
        const int64_t subsets_in_block = k == 0 ? 1 : int64_t(flat.size()) / k;
        for (int64_t s = 0; s < subsets_in_block; ++s) {
            // the equalities and the chosen rows as a square system
            Rows system;
            Row rhs;
            for (const std::size_t i : independent) {
                system.push_back(M.Ae[i]);
                rhs.push_back(M.be[i]);
            }
            for (int64_t j = 0; j < k; ++j) {
                system.push_back(M.A[std::size_t(flat[std::size_t(s * k + j)])]);
                rhs.push_back(M.b[std::size_t(flat[std::size_t(s * k + j)])]);
            }
            // its solution is a vertex if it satisfies all rows
            const std::optional<Row> x = priv_solve(system, rhs, numerics::kPolytopeRank);
            if (x && aux_feasible(M, *x) && !aux_listed(vertices, *x)) vertices.push_back(*x);
        }
        if (k == 0) break;
        flat = subsets.next(4096);
    } while (!flat.empty());
    return vertices;
}

// Halfspaces of points ----------------------------------------------------------------------------

HostH priv_constraints_of_points(const Tensor &like, const Rows &points, int64_t n) {
    HostH H;
    H.n = n;
    // a single vertex: {x | I x = v}
    if (points.size() == 1) {
        for (int64_t i = 0; i < n; ++i) {
            Row e(std::size_t(n), 0.0);
            e[std::size_t(i)] = 1.0;
            H.Ae.push_back(e);
            H.be.push_back(points[0][std::size_t(i)]);
        }
        return H;
    }
    if (n == 1) return aux_constraints_1d(points);
    const int64_t count = int64_t(points.size());
    if (n == 2) {
        // fewer than three vertices, or a rank-one matrix of them: a segment
        std::vector<double> flat(std::size_t(2 * count));
        for (int64_t k = 0; k < count; ++k) {
            flat[std::size_t(k)] = points[std::size_t(k)][0];
            flat[std::size_t(count + k)] = points[std::size_t(k)][1];
        }
        if (count <= 2 || Tensor::like(like, flat, {2, count}).matrix_rank() < 2)
            return aux_constraints_2d_degenerate(points);
    }
    // the affine hull: center, then the singular vectors of the centered points
    Row c(std::size_t(n), 0.0);
    for (const Row &p : points)
        for (int64_t i = 0; i < n; ++i) c[std::size_t(i)] += p[std::size_t(i)];
    for (double &x : c) x /= double(count);
    const int64_t width = std::max(count, n);
    std::vector<double> centered(std::size_t(n * width), 0.0);
    for (int64_t k = 0; k < count; ++k)
        for (int64_t i = 0; i < n; ++i)
            centered[std::size_t(i * width + k)] =
                points[std::size_t(k)][std::size_t(i)] - c[std::size_t(i)];
    const auto svd = Tensor::like(like, centered, {n, width}).svd();
    const std::vector<double> U = svd[0].data(), S = svd[1].data();
    const int64_t columns = svd[0].shape()[1];
    int64_t r = 0;
    for (const double s : S)
        if (std::abs(s) > numerics::kPolytopeSingular) ++r;
    if (r == 0) {
        // all vertices coincide
        for (int64_t i = 0; i < n; ++i) {
            Row e(std::size_t(n), 0.0);
            e[std::size_t(i)] = 1.0;
            H.Ae.push_back(e);
            H.be.push_back(c[std::size_t(i)]);
        }
        return H;
    }
    // the points in the coordinates of the hull (the singular vectors of the first r values)
    Rows y(static_cast<std::size_t>(count), Row(static_cast<std::size_t>(r)));
    for (int64_t k = 0; k < count; ++k)
        for (int64_t j = 0; j < r; ++j) {
            double sum = 0.0;
            for (int64_t i = 0; i < n; ++i)
                sum += (r == n ? (i == j ? 1.0 : 0.0) : U[std::size_t(i * columns + j)]) *
                       (points[std::size_t(k)][std::size_t(i)] - c[std::size_t(i)]);
            y[std::size_t(k)][std::size_t(j)] = sum;
        }
    for (const auto &[a, d] : aux_facets(y, std::size_t(r))) {
        // back to the ambient space: U [a; 0], and the offset of the center
        Row row(std::size_t(n), 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < r; ++j)
                row[std::size_t(i)] +=
                    (r == n ? (i == j ? 1.0 : 0.0) : U[std::size_t(i * columns + j)]) *
                    a[std::size_t(j)];
        H.A.push_back(row);
        H.b.push_back(d + priv_dot(row, c));
    }
    // the directions the points do not span are equalities
    for (int64_t j = r; j < n; ++j) {
        Row e(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) e[std::size_t(i)] = U[std::size_t(i * columns + j)];
        H.Ae.push_back(e);
        H.be.push_back(priv_dot(e, c));
    }
    return H;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
