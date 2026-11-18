// contains - whether a zonotope contains a point, a zonotope or a box, as CORA's contains_
//
// Exact: a linear program per point; for a set, the vertex search ("exact:venum") or the support
// of S along each facet normal of Z ("exact:polymax"); "exact" picks polymax below dimension 4, as
// CORA does. "approx" is the sufficient test of Sadraddini and Tedrake; "approx:stDual", "sampling"
// and "opt" are not supported. A zonotope that is not full-dimensional is enlarged by tol. Not for
// a batch.
//
// Syntax:   res = Z.contains(p);   res = Z.contains(S, "approx", tol);
// Inputs:   p, S, I - points (n, N), a zonotope or a box of the dimension of Z
//           method - "exact" (default), "exact:venum", "exact:polymax", "approx" or "approx:st"
//           tol - relaxes the points' norm and the flatness of Z; default: the repository tolerance
// Outputs:  res - true if all points (S) are in Z; "approx" says true only when certain
// See also: boundary_point, Interval::contains
//
// CORA: contSet/@zonotope/contains_.m
// CORA: matrixSet/@matZonotope/contains.m
// CORA: contSet/@zonotope/private/priv_zonotopeContainment_pointContainment.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Adrian Kulmburg, Ivan Brkan
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv_stacking.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Points ------------------------------------------------------------------------------------------

/// Whether the points (n x N, column by column on the host) are in the zonotope: the norm of
/// p - c is at most 1 (CORA's priv_zonotopeContainment_pointContainment, 'exact:venum').
bool aux_contains_points(const HostZonotope &Z, const std::vector<double> &p, int64_t N,
                         double tol) {
    for (int64_t k = 0; k < N; ++k) {
        std::vector<double> shifted(Z.n);
        for (int64_t i = 0; i < Z.n; ++i) shifted[i] = p[i * N + k] - Z.c[i];
        const double scaling = priv_zonotope_norm(Z, shifted);
        if (!(scaling <= 1 || within_tol(scaling, 1, tol))) return false;
    }
    return true;
}

/// The zonotope as CORA's contains_ prepares it: without zero generators, and enlarged by tol if
/// it is not full-dimensional.
Zonotope aux_prepared(const Zonotope &Z, double tol) {
    const Zonotope compacted = Z.compact();
    const int64_t m = compacted.nr_generators();
    if (m > 0 && compacted.generators().matrix_rank(tol) == Z.dim()) return compacted;
    const Tensor c = compacted.center().expand_dims(-1);
    const Tensor r = c.full_like(c.shape(), tol);
    return compacted.plus(Interval(r.negative(), r));
}

/// Whether the zonotope contains the points of the matrix p (n, N); `exact` picks the program.
bool aux_contains_matrix(const Zonotope &Z, const Tensor &p, double tol) {
    const int64_t N = p.shape().back();
    const Zonotope compacted = Z.compact();
    if (compacted.nr_generators() == 0) { // a point
        const std::vector<double> c = compacted.center().data(), x = p.data();
        for (int64_t i = 0; i < Z.dim(); ++i)
            for (int64_t k = 0; k < N; ++k)
                if (!within_tol(x[i * N + k], c[i], tol)) return false;
        return true;
    }
    if (priv_is_box(compacted, tol)) { // a box: every point by its bounds
        const Interval box = compacted.interval();
        for (int64_t k = 0; k < N; ++k)
            if (!box.contains(p.slice(-1, k, k + 1), tol)) return false;
        return true;
    }
    return aux_contains_points(priv_host_zonotope(aux_prepared(Z, tol)), p.data(), N, tol);
}

// Sets --------------------------------------------------------------------------------------------

/// Whether the support of S along each halfspace C x <= d of Z is at most d (CORA's contains_ of
/// polytope(Z) by 'exact:polymax'): one matrix product, fast for many generators in low dimension.
bool aux_contains_polymax(const Zonotope &Z, const Zonotope &S, double tol) {
    const auto [C, d] = priv_zonotope_halfspaces(Z);
    // layout: the product with CORA's columns
    const Tensor support = C.matmul(S.center().expand_dims(-1)) +
                           C.matmul(S.generators().matrix_transpose()).abs_sum();
    const std::vector<double> b = d.data(), b_ = support.data();
    for (size_t i = 0; i < b.size(); ++i)
        if (b_[i] > b[i] && !within_tol(b[i], b_[i], tol)) return false;
    return true;
}

/// Whether the zonotope contains the zonotope S by the method.
bool aux_contains_zonotope(const Zonotope &Z, const Zonotope &S, const std::string &method,
                           double tol) {
    const Zonotope inner = S.compact();
    if (inner.nr_generators() == 0)
        return aux_contains_matrix(Z, inner.center().expand_dims(-1), tol);
    const Zonotope compacted = Z.compact();
    if (compacted.nr_generators() == 0) return false; // a point holds no set
    if (priv_is_box(compacted, tol)) { // a box contains S iff it contains the box around S
        const Interval box = compacted.interval(), around = inner.interval();
        return box.contains(around.inf(), tol) && box.contains(around.sup(), tol);
    }
    const Zonotope outer = aux_prepared(Z, tol);
    // "exact" is the vertex search from dimension 4 on, as CORA's aux_exactParser decides
    const bool polymax = method == "exact:polymax" || (method == "exact" && Z.dim() < 4);
    if (polymax) return aux_contains_polymax(outer, inner, tol);
    if (method == "exact" || method == "exact:venum")
        return priv_containment_vertex_enumeration(priv_host_zonotope(inner),
                                                   priv_host_zonotope(outer), tol);
    const double scaling =
        priv_containment_sadraddini_tedrake(priv_host_zonotope(inner), priv_host_zonotope(outer));
    return scaling <= 1 + tol;
}

/// Why the call is refused, if it is: a batch, a method not known for the operand (points or a
/// set) or not supported.
std::optional<std::pair<Error::Id, std::string>> aux_refusal(bool batch, const std::string &method,
                                                             bool points) {
    using Reason = std::pair<Error::Id, std::string>;
    if (batch)
        return Reason{Error::not_supported, "a linear program per set: it supports a single "
                                            "zonotope, not a batch"};
    const bool for_sets = method == "approx:st" || method == "approx:stDual" ||
                          method == "sampling" || method == "sampling:primal" ||
                          method == "sampling:dual" || method == "opt";
    const bool known = method == "exact" || method == "exact:venum" || method == "exact:polymax" ||
                       method == "approx" || (!points && for_sets);
    if (!known) return Reason{Error::no_specific_alg, "no algorithm '" + method + "' for this set"};
    if (method == "approx:stDual" || method == "opt" || method.rfind("sampling", 0) == 0)
        return Reason{Error::not_supported, "the method '" + method + "' is not supported"};
    return std::nullopt;
}

// Matrices ----------------------------------------------------------------------------------------

/// Whether the matrices M (rows, cols) or (p, rows, cols) are in the matrix zonotope Z: their
/// stacked columns in the stacked zonotope (CORA's matZonotope.contains), type "exact" or "approx".
bool aux_contains_matrices(const Zonotope &Z, const Tensor &M, const std::string &type,
                           double tol) {
    // the points are the columns of the stacked matrices
    const Tensor matrices = M.ndim() == 2 ? M.expand_dims(0) : M;
    const Tensor points = priv_stack_columns(matrices).squeeze(-1).matrix_transpose();
    return Z.stacked().contains(points, type, tol);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    // matrices in a matrix zonotope: one set, matrices of its shape, CORA's types
    if (elem_rank_ == 2) {
        CORA_CHECK(method == "exact" || method == "approx", Error::wrong_value, [&] {
            return "contains: the type must be 'exact' or 'approx', got '" + method + "'";
        });
        CORA_CHECK(c_.ndim() == 2, Error::not_supported,
                   [] { return std::string("contains: not supported for a batch of sets"); });
        // matrices of the shape of the set, one or a list
        const int64_t rows = c_.size(-2), cols = c_.size(-1);
        CORA_CHECK((p.ndim() == 2 || p.ndim() == 3) && p.numel() > 0 && p.size(-2) == rows &&
                       p.size(-1) == cols,
                   Error::dimension_mismatch, [&] {
                       return "contains: the matrices must be (" + std::to_string(rows) + ", " +
                              std::to_string(cols) + ") or (p, rows, cols), got the shape " +
                              shape_text(p.shape());
                   });
        return aux_contains_matrices(*this, p, method, tol);
    }
    // points of a vector zonotope: columns (n, N)
    CORA_CHECK(p.ndim() == 2 && p.shape()[0] == dim(), Error::dimension_mismatch, [&] {
        return "contains: the points must be a matrix (" + std::to_string(dim()) +
               ", N) of columns, got the shape " + shape_text(p.shape());
    });
    const auto refusal = aux_refusal(c_.ndim() != 1, method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_contains_matrix(*this, p, tol);
}

bool Zonotope::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    // the dimensions, then the batch and the method
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "contains: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    const auto refusal = aux_refusal(c_.ndim() != 1 || S.center().ndim() != 1, method, false);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_contains_zonotope(*this, S, method, tol);
}

bool Zonotope::contains(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    CORA_CHECK(I.dim() == dim() && I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "contains: the interval must be a box of the dimension " + std::to_string(dim()) +
               " of the zonotope, got bounds of the shape " + shape_text(I.inf().shape());
    });
    const auto refusal = aux_refusal(c_.ndim() != 1 || I.inf().ndim() != 2, method, false);

    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    // a box is the zonotope of its axes
    return aux_contains_zonotope(*this, I.zonotope(), method, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
