// isequal - whether a polytope equals a polytope, points or another set, as CORA's polytope.isequal
//
// A different dimension is false. R^n equals only R^n. In one dimension the vertices decide. Two
// vertex sets are equal if their extreme vertices agree up to order. Two full-dimensional sets are
// equal if every non-redundant row of one is a row of the other (the rows normalized to unit
// length); else (and for any other set) each set must contain the other. A column of points is
// equal to a set that is that point. CORA's own tolerance is 1e-6; here the default is the
// repository's. Not a batch.
//
// Syntax:   res = P.isequal(S);   res = P.isequal(S, tol);   res = P.isequal(p, tol);
// Inputs:   S - a polytope, or a box, zonotope, constrained zonotope, bundle, empty set, R^n
//           p - a point (n, 1)
//           tol - what counts as equal (as within_tol); default: the repository tolerance
// Outputs:  res - whether both describe the same set
// See also: Polytope::contains, Polytope::compact
//
// CORA: contSet/@polytope/isequal.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Comparisons of vertices -------------------------------------------------------------------------

/// Whether the columns of M1 and M2 (n x k each) agree up to order: each column of M1 takes the
/// first unused column of M2 that equals it up to tol (CORA's compareMatrices).
bool aux_same_columns(const Rows &M1, const Rows &M2, double tol) {
    if (M1.size() != M2.size()) return false;
    std::vector<bool> used(M2.size(), false);
    for (const Row &a : M1) {
        bool found = false;
        for (std::size_t j = 0; j < M2.size() && !found; ++j) {
            bool equal = !used[j];
            for (std::size_t i = 0; equal && i < a.size(); ++i)
                equal = within_tol(a[i], M2[j][i], tol);
            if (equal) found = used[j] = true;
        }
        if (!found) return false;
    }
    return true;
}

/// The bounds of a one-dimensional set as its vertices (a point once), -Inf and Inf included.
Row aux_bounds_1d(const Polytope &P) {
    Row v;
    for (const Row &p : priv_polytope_vertices(P)) v.push_back(p[0]);
    return v;
}

/// Whether the one-dimensional vertex lists agree (CORA's aux_isequal_1D): infinite ends equal.
bool aux_equal_1d(const Row &v1, const Row &v2, double tol) {
    const auto infinite = [](const Row &v) {
        return std::any_of(v.begin(), v.end(), [](double x) { return std::isinf(x); });
    };
    if (infinite(v1) != infinite(v2) || v1.size() != v2.size()) return false;
    if (infinite(v1)) return v1 == v2;
    Rows a, b;
    for (const double x : v1) a.push_back({x});
    for (const double x : v2) b.push_back({x});
    return aux_same_columns(a, b, tol);
}

// Full-dimensional sets by their rows -------------------------------------------------------------

/// The rows [a, b] of the inequalities of H with unit normals.
Rows aux_unit_rows(HostH H) {
    priv_normalize_rows(H);
    auto [A, b] = priv_equality_to_inequality(H);
    Rows rows;
    for (std::size_t i = 0; i < A.size(); ++i) {
        Row row = A[i];
        row.push_back(b[i]);
        rows.push_back(row);
    }
    return rows;
}

/// Whether row j of the rows [a, b] is not implied by the others (CORA's
/// aux_isNonredundantConstraint): the others reach beyond b_j along a_j.
bool aux_nonredundant(const Rows &rows, std::size_t j, double tol) {
    HostH others;
    others.n = int64_t(rows[j].size()) - 1;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i == j) continue;
        others.A.emplace_back(rows[i].begin(), rows[i].end() - 1);
        others.b.push_back(rows[i].back());
    }
    const Row direction(rows[j].begin(), rows[j].end() - 1);
    const Support s = priv_support(others, direction, true);
    const double reach = s.status == Support::Status::infeasible ? -kInf : s.value;
    return reach > rows[j].back() && !within_tol(reach, rows[j].back(), tol);
}

/// Whether every row of `rows` that no row of `other` repeats is redundant in `rows`.
bool aux_rows_covered(const Rows &rows, const Rows &other, double tol) {
    for (std::size_t j = 0; j < rows.size(); ++j) {
        const bool found = std::any_of(other.begin(), other.end(), [&](const Row &r) {
            for (std::size_t k = 0; k < r.size(); ++k)
                if (!within_tol(r[k], rows[j][k], tol)) return false;
            return true;
        });
        if (!found && aux_nonredundant(rows, j, tol)) return false;
    }
    return true;
}

/// Equality of two full-dimensional sets by their non-redundant rows.
bool aux_equal_rows(const Polytope &P, const Polytope &S, double tol) {
    const Rows rows_P = aux_unit_rows(priv_host_h(P.constraints()));
    const Rows rows_S = aux_unit_rows(priv_host_h(S.constraints()));
    return aux_rows_covered(rows_P, rows_S, tol) && aux_rows_covered(rows_S, rows_P, tol);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Polytopes ---------------------------------------------------------------------------------------

bool Polytope::isequal(const Polytope &S, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "isequal");
    CORA_CHECK(like.ndim() == 2 && (S.is_h_rep() ? S.A() : S.V()).ndim() == 2, Error::not_supported,
               [] { return std::string("isequal: it supports single polytopes, not a batch"); });
    if (S.dim() != dim()) return false;
    // R^n is equal to R^n only
    if (priv_is_fullspace(*this)) return priv_is_fullspace(S);
    if (priv_is_fullspace(S)) return false;
    if (dim() == 1) return aux_equal_1d(aux_bounds_1d(*this), aux_bounds_1d(S), tol);
    // two vertex sets: their extreme vertices up to order
    if (is_v_rep() && S.is_v_rep())
        return aux_same_columns(priv_polytope_vertices(*this), priv_polytope_vertices(S),
                                numerics::kPolytopeVertexMerge);
    const bool full = is_full_dim(), full_S = S.is_full_dim();
    if (full != full_S) return false;
    if (full) return aux_equal_rows(*this, S, tol);
    return contains(S, "exact", tol) && S.contains(*this, "exact", tol);
}

// Points and other sets ---------------------------------------------------------------------------

bool Polytope::isequal(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "isequal");
    CORA_CHECK(is_column(p, dim()) && p.ndim() == 2 && like.ndim() == 2, Error::dimension_mismatch,
               [&] {
                   return "isequal: a point must be a column (" + std::to_string(dim()) +
                          ", 1) and the polytope single";
               });
    // the set is that point: one vertex, equal up to tol
    if (!representsa("point", tol)) return false;
    const Rows vertices = priv_polytope_vertices(*this);
    const Row x = p.data();
    return std::all_of(vertices.begin(), vertices.end(), [&](const Row &v) {
        for (std::size_t i = 0; i < x.size(); ++i)
            if (!within_tol(v[i], x[i], tol)) return false;
        return true;
    });
}

bool Polytope::isequal(const ContSet &S, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "isequal");
    if (S.dim() != dim()) return false;
    switch (S.class_id()) {
    case ClassId::polytope:
        return isequal(static_cast<const Polytope &>(S), tol);
    // the two special sets are decided by what the polytope represents
    case ClassId::empty_set:
        return representsa("emptySet", tol);
    case ClassId::fullspace:
        return representsa("fullspace", tol);
    default:
        break;
    }
    // a set of lower precedence decides itself
    CORA_CHECK(like.ndim() == 2, Error::not_supported,
               [] { return std::string("isequal: it supports a single polytope, not a batch"); });
    if (precedence(S.class_id()) < precedence(class_id())) return cora::isequal(S, *this, tol);
    return contains(S, "exact", tol) && less_equal(S, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
