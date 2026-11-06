// is_full_dim - whether a polytope has a nonempty interior, as CORA's polytope.isFullDim
//
// A vertex polytope is full-dimensional if its centered vertices have rank n; a one-dimensional
// halfspace set if it has two vertices (one at infinity counts); a halfspace set without equalities
// if no pair of opposite rows is flat or empty and the largest slack of all rows is positive. The
// subspace (a basis of the directions of the affine hull) of a halfspace set is found by one linear
// program per axis and step, as CORA does. Not a batch.
//
// Syntax:   bool res = P.is_full_dim();   auto [res, X] = P.is_full_dim_subspace(tol);
// Inputs:   tol - the rank tolerance and that of flat opposite rows (default: repository's)
// Outputs:  res - whether the interior is nonempty
//           X - a basis (n, k) of the span of the set: the identity if res, else k < n columns
// See also: Polytope::representsa, Polytope::rand_point
//
// CORA: contSet/@polytope/isFullDim.m
// CORA: contSet/@polytope/private/priv_isFullDim_V.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The halfspace tests -----------------------------------------------------------------------------

/// The vertices of a one-dimensional set as numbers: none if empty.
Row aux_vertices_1d(const Polytope &P) {
    bool empty = false;
    return priv_vertices_1d(priv_host_h(P.constraints()), empty);
}

/// Whether the rows have an anti-parallel pair that is flat or empty (CORA's test on the
/// normalized rows): a x <= b and -a x <= c with -c >= b.
bool aux_flat_pair(HostH H, double tol) {
    priv_normalize_rows(H);
    for (std::size_t r = 0; r < H.A.size(); ++r) {
        for (std::size_t c = 0; c < r; ++c) {
            if (!within_tol(priv_dot(H.A[r], H.A[c]), -1.0, machine_eps())) continue;
            if (-H.b[c] > H.b[r] || within_tol(-H.b[c], H.b[r], tol)) return true;
        }
    }
    return false;
}

/// Whether a halfspace set without equalities has an interior: the largest common slack t of
/// A x + t <= b is positive (CORA's aux_isFullDim_nD_Hpoly_nosubspace).
bool aux_slack_positive(const HostH &H) {
    LinProg lp;
    lp.n = int(H.n + 1);
    lp.f.assign(std::size_t(lp.n), 0.0);
    lp.f[std::size_t(H.n)] = -1.0;
    for (Row row : H.A) {
        row.push_back(1.0);
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
    }
    lp.bineq = H.b;
    lp.lb.assign(std::size_t(lp.n), -kInf);
    lp.lb[std::size_t(H.n)] = 0.0;
    const LinProgResult r = linprog(lp);
    // no point: empty; no bound on the slack: an unbounded set with interior
    if (r.status == LinProgResult::Status::infeasible) return false;
    if (r.status == LinProgResult::Status::unbounded) return true;
    return !within_tol(0.0, r.fval, numerics::kPolytopeFullDim);
}

// The subspace ------------------------------------------------------------------------------------

/// The farthest the set reaches along +-e_i in the part orthogonal to the span X, and where: the
/// halfspaces in the variables (x, lambda) with z = x + X lambda and X'x = 0 (CORA's
/// aux_maxNormPerpendicularPolytope); none if every extent is zero.
std::optional<Row> aux_perpendicular_extent(const HostH &H, const Rows &X) {
    const std::size_t n = std::size_t(H.n), k = X.size();
    HostH ext{H.n + int64_t(k), {}, H.b, {}, H.be};
    const auto widen = [&](const Row &a) {
        Row row = a;
        for (const Row &column : X) row.push_back(priv_dot(a, column));
        return row;
    };
    for (const Row &a : H.A) ext.A.push_back(widen(a));
    for (const Row &a : H.Ae) ext.Ae.push_back(widen(a));
    for (const Row &column : X) {
        Row row = column;
        row.resize(n + k, 0.0);
        ext.Ae.push_back(row);
        ext.be.push_back(0.0);
    }
    // the extent along every signed axis, by one program each
    double best = 0.0;
    std::optional<Row> maximizer;
    for (std::size_t i = 0; i < 2 * n; ++i) {
        Row y(n + k, 0.0);
        y[i % n] = i < n ? 1.0 : -1.0;
        Support s = priv_support(ext, y, true);
        // an unbounded extent: fix the value 1 of the objective to get a point
        if (s.status == Support::Status::unbounded) {
            HostH fixed = ext;
            fixed.Ae.push_back(y);
            fixed.be.push_back(1.0);
            s = priv_support(fixed, y, true);
            s.value = kInf;
        }
        if (s.status == Support::Status::infeasible || s.value <= best) continue;
        best = s.value;
        maximizer = Row(s.x.begin(), s.x.begin() + std::ptrdiff_t(n));
    }
    return maximizer;
}

/// The affine span of a nonempty set (CORA's aux_isFullDim_nD_H_subspace): shift to a point of
/// the set, then add the direction of the farthest extent orthogonal to the span so far.
Rows aux_halfspace_span(const Polytope &P) {
    HostH H = priv_host_h(P.constraints());
    const std::optional<Row> x0 = priv_chebyshev_center(H);
    if (!x0 || std::isnan((*x0)[0])) return {};
    // the set shifted by -x0: the offsets lose A x0
    for (std::size_t i = 0; i < H.A.size(); ++i) H.b[i] -= priv_dot(H.A[i], *x0);
    for (std::size_t i = 0; i < H.Ae.size(); ++i) H.be[i] -= priv_dot(H.Ae[i], *x0);
    Rows span;
    for (int64_t i = 0; i < H.n; ++i) {
        const std::optional<Row> x = aux_perpendicular_extent(H, span);
        double largest = 0.0;
        if (x)
            for (const double v : *x) largest = std::max(largest, std::abs(v));
        if (largest <= numerics::kPolytopeSubspace) break;
        const double norm = priv_norm(*x);
        Row unit = *x;
        for (double &v : unit) v /= norm;
        span.push_back(unit);
    }
    return span;
}

// The vertex form ---------------------------------------------------------------------------------

/// The basis of the span of the centered vertices (CORA's priv_isFullDim_V): the left singular
/// vectors of the nonzero singular values.
std::pair<bool, Tensor> aux_vertex_span(const Tensor &V, double tol) {
    const int64_t n = V.shape()[0];
    // a line has an interior if its two ends differ (an end at infinity does)
    if (n == 1) {
        const bool full = V.numel() > 0 && !within_tol(V.min().item(), V.max().item(), tol);
        return {full, full ? priv_identity(V, 1) : priv_zeros(V, 1, 0)};
    }
    const Tensor centered = V - V.mean(-1, true);
    const int64_t rank = centered.shape().back() == 0 ? 0 : centered.matrix_rank(tol);
    if (rank == n) return {true, priv_identity(V, n)};
    if (rank == 0) return {false, priv_zeros(V, n, 0)};
    return {false, centered.svd()[0].slice(-1, 0, rank)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<bool, Tensor> Polytope::is_full_dim_subspace(Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "is_full_dim");
    CORA_CHECK(like.ndim() == 2, Error::not_supported, [] {
        return std::string("is_full_dim: it supports a single polytope, not a batch");
    });
    const int64_t n = dim();
    if (is_v_rep()) return aux_vertex_span(V_, tol);
    if (n == 1) {
        // two vertices (or a half-line) have an interior, a point or nothing has none
        const bool full = aux_vertices_1d(*this).size() == 2;
        return {full, full ? priv_identity(A_, 1) : priv_zeros(A_, 1, 0)};
    }
    if (priv_polytope_is_empty(*this)) return {false, priv_zeros(A_, n, 0)};
    const Rows span = aux_halfspace_span(*this);
    if (int64_t(span.size()) == n) return {true, priv_identity(A_, n)};
    // the vectors of the span are the columns of the basis
    return {false, priv_points_tensor(A_, n, span)};
}

bool Polytope::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "is_full_dim");
    if (is_v_rep() || dim() == 1) return is_full_dim_subspace(tol).first;
    CORA_CHECK(like.ndim() == 2, Error::not_supported, [] {
        return std::string("is_full_dim: it supports a single polytope, not a batch");
    });
    // equalities leave no interior; else opposite rows must not touch and some slack must remain
    const HostH H = priv_host_h(constraints());
    return H.Ae.empty() && !aux_flat_pair(H, tol) && aux_slack_positive(H);
}

bool Polytope::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
