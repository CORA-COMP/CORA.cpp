// center - a point near the middle of a polytope, as CORA's polytope.center
//
// "chebyshev": the center of the largest ball in the set, from the halfspaces (a linear program;
// for equalities alone their solution); "avg": the mean of the extreme vertices. R^n gives the
// origin; the empty set a column without entries (n, 0); a set whose Chebyshev radius is
// unbounded, or whose equalities leave a freedom, gives NaN. In 1D the middle of the two
// vertices. Not a batch, except for "avg" of the vertices of a single set.
//
// Syntax:   Tensor c = P.center();   Tensor c = P.center("avg");
// Inputs:   P - polytope;  method - "chebyshev" (default) or "avg"
// Outputs:  c - a column (n, 1)
// See also: interval, Polytope::contains
//
// CORA: contSet/@polytope/center.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Pieces of the center ----------------------------------------------------------------------------

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The tensor of the numbers as a column on the backend of the polytope's data.
Tensor aux_column(const Polytope &P, const Row &values) {
    return priv_column_tensor(P.is_h_rep() ? P.A() : P.V(), values);
}

/// The center of a one-dimensional polytope from its vertices: the vertex, the middle of two, NaN
/// if one is infinite, nothing if there is none.
Tensor aux_center_1d(const Polytope &P) {
    Row v;
    if (P.is_v_rep()) {
        for (const Row &point : priv_host_points(P.V())) v.push_back(point[0]);
    } else {
        bool empty = false;
        v = priv_vertices_1d(priv_host_h(P.constraints()), empty);
    }
    const Tensor like = P.is_h_rep() ? P.A() : P.V();
    if (v.empty()) return priv_zeros(like, 1, 0);
    if (v.size() == 1) return aux_column(P, {v[0]});
    if (std::isinf(v[0]) || std::isinf(v[1])) return aux_column(P, {kNaN});
    return aux_column(P, {(v[0] + v[1]) / 2.0});
}

/// The mean of the extreme vertices.
Tensor aux_center_avg(const Polytope &P) {
    if (P.is_v_rep()) {
        const Rows points = priv_host_points(P.V());
        // of several vertices only the extreme ones count
        const Tensor V = points.size() > 1 ? P.V().take(priv_extreme_points(points), -1) : P.V();
        return V.mean(-1, true);
    }
    const Rows vertices = priv_polytope_vertices(P);
    const int64_t n = P.dim();
    if (vertices.empty()) return priv_zeros(P.A(), n, 0);
    Row mean(std::size_t(n), 0.0);
    for (const Row &v : vertices)
        for (int64_t i = 0; i < n; ++i) mean[std::size_t(i)] += v[std::size_t(i)];
    for (double &x : mean) x /= double(vertices.size());
    return aux_column(P, mean);
}

/// The solution of the equalities alone: the point if there is exactly one, NaN if they leave a
/// freedom, nothing if they contradict each other (by elimination with partial pivoting).
std::optional<Row> aux_equalities_only(const HostH &H) {
    Rows M = H.Ae;
    Row r = H.be;
    const std::size_t rows = M.size(), n = std::size_t(H.n);
    double scale = 1.0;
    for (const Row &row : M)
        for (const double x : row) scale = std::max(scale, std::abs(x));
    std::vector<std::size_t> pivot_column;
    std::size_t done = 0;
    for (std::size_t c = 0; c < n && done < rows; ++c) {
        std::size_t pivot = done;
        for (std::size_t i = done + 1; i < rows; ++i)
            if (std::abs(M[i][c]) > std::abs(M[pivot][c])) pivot = i;
        if (std::abs(M[pivot][c]) <= numerics::kPolytopeRank * scale) continue;
        std::swap(M[done], M[pivot]);
        std::swap(r[done], r[pivot]);
        for (std::size_t i = 0; i < rows; ++i) {
            if (i == done) continue;
            const double f = M[i][c] / M[done][c];
            for (std::size_t j = 0; j < n; ++j) M[i][j] -= f * M[done][j];
            r[i] -= f * r[done];
        }
        pivot_column.push_back(c);
        ++done;
    }
    // a row without coefficients but with an offset contradicts the others
    for (std::size_t i = done; i < rows; ++i)
        if (std::abs(r[i]) > numerics::kPolytopeRank * scale) return std::nullopt;
    if (done < n) return Row(n, kNaN);
    Row x(n);
    for (std::size_t i = 0; i < n; ++i) x[pivot_column[i]] = r[i] / M[i][pivot_column[i]];
    return x;
}

/// The Chebyshev center of the halfspaces of P: the point, NaN, or nothing for the empty set.
Tensor aux_center_chebyshev(const Polytope &P) {
    const HostH H = priv_host_h(P.constraints());
    const std::optional<Row> c =
        H.A.empty() && !H.Ae.empty() ? aux_equalities_only(H) : priv_chebyshev_center(H);
    if (!c) return priv_zeros(P.is_h_rep() ? P.A() : P.V(), H.n, 0);
    return aux_column(P, *c);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The center --------------------------------------------------------------------------------------

Tensor Polytope::center_impl() const { return center("chebyshev"); }

Tensor Polytope::center(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(method == "chebyshev" || method == "avg", Error::wrong_value, [&] {
        return "center: the method must be 'chebyshev' or 'avg', got '" + method + "'";
    });
    const Tensor &like = is_h_rep() ? A_ : V_;
    const int64_t n = dim();
    // R^n has the origin as its center, a set without vertices none
    if (priv_is_fullspace(*this)) return priv_zeros(like, n, 1);
    if (priv_has_no_vertices(*this)) return priv_zeros(like, n, 0);
    if (n == 1) return aux_center_1d(*this);
    return method == "avg" ? aux_center_avg(*this) : aux_center_chebyshev(*this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
