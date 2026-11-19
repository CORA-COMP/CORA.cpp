// contains - whether a polytope contains points or a set, as CORA's polytope.contains
//
// "exact:venum" uses the vertices of P, "exact:polymax" the halfspaces; "approx" is "exact" for a
// box or a zonotope and not supported for others; S is any set with a support function. A
// halfspace polytope holds a point if every row is violated by at most tol (absolute), and a set
// if the support function of the set along every row (an equality both ways) is at most the
// offset up to tol (within_tol); a polytope with vertices counts as its points. A vertex polytope
// holds a point if it is a vertex up to tol or a convex combination of the vertices (a linear
// program); a set is tested on the halfspaces of P. R^n holds everything and holds only itself;
// the empty set holds nothing but the empty set. In 1D a set is its two bounds. CORA's
// certificate and scaling outputs are not returned. Not a batch.
//
// Syntax:   res = P.contains(p);   res = P.contains(S, "exact", tol);
// Inputs:   p - points (..., n), one per row
//           S - a set of the dimension of P: a box, a zonotope, a polytope, ...
//           method - "exact" (default), "exact:venum" or "exact:polymax"
//           tol - see the description; default: the repository tolerance
// Outputs:  res - true if all points (the whole set) are in P
// See also: Polytope::intersect, ConZonotope::contains
//
// CORA: contSet/@polytope/contains_.m
// CORA authors: Niklas Kochdumper, Viktor Kotsev, Adrian Kulmburg, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Refusals and points -----------------------------------------------------------------------------

/// Why the method is refused, if it is.
detail::Refusal aux_method_refusal(const std::string &method) {
    if (method == "exact" || method == "exact:venum" || method == "exact:polymax" ||
        method == "approx")
        return std::nullopt;
    return std::pair{Error::no_specific_alg,
                     "contains: no algorithm '" + method + "' for this set"};
}

/// Whether every entry of a 0/1 mask is 1 (true for none).
bool aux_all(const Tensor &mask) { return mask.numel() == 0 || mask.min().item() == 1.0; }

/// Whether the points (N, n), one per row, satisfy the halfspaces up to tol: the offsets of the
/// rows are at most tol, those of the equalities at most tol in magnitude (A x on CORA's columns).
bool aux_points_in_halfspaces(const Polytope::Constraints &C, const Tensor &points, double tol) {
    const Tensor p = points.matrix_transpose();
    if (C.b.numel() > 0 &&
        !aux_all((C.A.matmul(p) - C.b.expand_dims(-1)).scalar_op(Tensor::Binary::less_equal, tol)))
        return false;
    return C.be.numel() == 0 || aux_all((C.Ae.matmul(p) - C.be.expand_dims(-1))
                                            .abs()
                                            .scalar_op(Tensor::Binary::less_equal, tol));
}

/// Whether the points (N, n) are in the convex hull of the vertices of a single polytope.
bool aux_points_in_vertices(const Rows &vertices, int64_t n, const Tensor &p, double tol) {
    const Rows points = priv_host_points(p);
    for (const Row &x : points) {
        bool vertex = false;
        for (const Row &v : vertices) {
            bool same = true;
            for (int64_t i = 0; i < n; ++i) same = same && within_tol(x[i], v[i], tol);
            vertex = vertex || same;
        }
        if (!vertex && !priv_in_hull(vertices, x)) return false;
    }
    return true;
}

/// Whether the points (N, n) are in P by the routes of CORA's point cloud: the halfspaces if P has
/// them, else the vertices.
bool aux_points_in_polytope(const Polytope &P, const Tensor &p, double tol) {
    if (P.is_h_rep()) return aux_points_in_halfspaces({P.A(), P.b(), P.Ae(), P.be()}, p, tol);
    if (P.dim() == 1) {
        const Tensor lo = P.V().min(), hi = P.V().max();
        return aux_all(p.greater_equal(lo).multiply(p.less_equal(hi)));
    }
    return aux_points_in_vertices(priv_host_points(P.V()), P.dim(), p, tol);
}

// Sets --------------------------------------------------------------------------------------------

/// The support function of a set along a direction (a host row).
double aux_support(const ContSet &S, const Tensor &like, const Row &direction) {
    if (S.class_id() == ClassId::polytope) {
        const Polytope &Q = static_cast<const Polytope &>(S);
        return priv_support(priv_host_h(Q.constraints()), direction, true).value;
    }
    const Tensor value = S.support_function(priv_vector_tensor(like, direction).expand_dims(-1));
    if (value.numel() != 1)
        CORA_FAIL(Error::not_supported, "contains: the set must be a single set, not a batch");
    return value.item();
}

/// Whether the set S is inside the halfspaces C of P: its support function along each row (an
/// equality both ways) is at most the offset.
bool aux_set_in_halfspaces(const Polytope::Constraints &C, const ContSet &S, double tol) {
    const HostH H = priv_host_h(C);
    const auto [rows, offsets] = priv_equality_to_inequality(H);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double bound = aux_support(S, C.A, rows[i]);
        if (bound > offsets[i] && !within_tol(offsets[i], bound, tol)) return false;
    }
    return true;
}

/// A polytope with vertices counts as the points of its vertices.
const Polytope *aux_vertex_polytope(const ContSet &S) {
    if (S.class_id() != ClassId::polytope) return nullptr;
    const Polytope &Q = static_cast<const Polytope &>(S);
    return Q.is_v_rep() ? &Q : nullptr;
}

/// Whether S is the empty set (a box without bounds, a polytope without points).
bool aux_is_empty(const ContSet &S) {
    if (S.class_id() == ClassId::empty_set) return true;
    if (S.class_id() == ClassId::interval) return static_cast<const Interval &>(S).is_empty();
    if (S.class_id() == ClassId::polytope)
        return priv_polytope_is_empty(static_cast<const Polytope &>(S));
    return false;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Points ------------------------------------------------------------------------------------------

bool Polytope::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "contains");
    const detail::Refusal refusal = aux_method_refusal(method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(p.defined() && is_vector(p, dim()), Error::dimension_mismatch, [&] {
        return "contains: the points must be (..., " + std::to_string(dim()) +
               "), one per row, got the shape " + (p.defined() ? shape_text(p.shape()) : "()");
    });
    if (priv_is_fullspace(*this)) return true;
    if (is_h_rep() && A_.ndim() != 2)
        CORA_FAIL(Error::not_supported, "contains: a single polytope is supported, not a batch");
    if (p.numel() == 0) return true;
    return aux_points_in_polytope(*this, p.reshape({p.numel() / dim(), dim()}), tol);
}

// Sets --------------------------------------------------------------------------------------------

bool Polytope::contains(const ContSet &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "contains");
    const detail::Refusal refusal = aux_method_refusal(method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "contains: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    if (like.ndim() != 2)
        CORA_FAIL(Error::not_supported, "contains: a single polytope is supported, not a batch");
    // R^n holds every set, and only R^n holds R^n
    if (priv_is_fullspace(*this)) return true;
    if (S.class_id() == ClassId::fullspace) return false;
    // the empty set is in every set, and no other set is in the empty set
    if (aux_is_empty(S)) return true;
    if (priv_polytope_is_empty(*this)) return false;
    if (method == "approx" && S.class_id() != ClassId::interval &&
        S.class_id() != ClassId::zonotope)
        CORA_FAIL(Error::not_supported, "contains: 'approx' needs a zonotope or a box");
    // in 1D a set is its two bounds
    if (dim() == 1) {
        const Interval hull = S.interval();
        return contains(Tensor::stack({hull.inf(), hull.sup()}, 0), "exact", tol);
    }
    // 'exact' goes by the form P has, 'exact:venum' by its vertices, 'exact:polymax' by its rows
    const Polytope *vertex_set = aux_vertex_polytope(S);
    const bool by_vertices = method == "exact:venum" || (method != "exact:polymax" && !is_h_rep());
    if (by_vertices && vertex_set) {
        const Rows vertices = priv_polytope_vertices(*this);
        return aux_points_in_vertices(vertices, dim(), vertex_set->V(), tol);
    }
    // all vertex polytopes are bounded, so an unbounded set is not in one
    if (by_vertices && S.class_id() == ClassId::polytope &&
        !priv_polytope_is_bounded(static_cast<const Polytope &>(S)))
        return false;
    const Constraints C = constraints();
    if (vertex_set) return aux_points_in_halfspaces(C, vertex_set->V(), tol);
    return aux_set_in_halfspaces(C, S, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
