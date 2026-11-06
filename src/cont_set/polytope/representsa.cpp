// representsa - whether a polytope is also a set of another CORA type, as CORA's representsa_
//
// The set is tested, not converted: the converted set is not returned. A polytope without
// constraints (R^n) or without vertices answers by CORA's table for an empty object. "zonotope"
// and "parallelotope" are not decided by CORA (Error::not_supported), "levelSet" and "convexSet"
// are always true. "interval" reads the rows with one nonzero entry as bounds, as CORA does. Not a
// batch.
//
// Syntax:   bool res = P.representsa("point");   bool res = P.representsa("fullspace", tol);
// Inputs:   type - a type of set, CORA's name ("origin", "point", "emptySet", "interval", ...)
//           tol - the tolerance of the tests; default: the repository tolerance
// Outputs:  res - whether the set is a set of that type
// See also: Polytope::is_bounded, Polytope::is_full_dim, Polytope::contains
//
// CORA: contSet/@polytope/representsa_.m
// CORA: contSet/@polytope/isemptyobject.m
// CORA: contSet/@contSet/representsa_emptyObject.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_set_type.h"
#include "cont_set/polytope/private/priv.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The empty object --------------------------------------------------------------------------------

/// Whether the polytope has no constraint and no vertex in the forms it holds (CORA's
/// isemptyobject): R^n in halfspace form, or a set without vertices.
bool aux_empty_object(const Polytope &P) {
    const bool no_rows = !P.is_h_rep() || (P.b().numel() == 0 && P.be().numel() == 0);
    const bool no_vertices = !P.is_v_rep() || P.V().numel() == 0;
    return no_rows && no_vertices;
}

/// The answer for an empty object, by CORA's table (representsa_emptyObject).
bool aux_empty_object_answer(const Polytope &P, const std::string &type) {
    if (type == "polytope" || type == "hyperplane" || type == "interval") return true;
    if (type == "origin" || type == "point" || type == "parallelotope") return false;
    if (type == "fullspace") return P.is_h_rep();
    return P.dim() == 0 || (P.is_v_rep() && P.V().numel() == 0);
}

// Tests of the set --------------------------------------------------------------------------------

/// The support function of the box (lb, ub) along a, as an Interval gives it.
double aux_box_support(const Tensor &like, const Row &lb, const Row &ub, const Row &a, bool upper) {
    const Interval box(priv_column_tensor(like, lb), priv_column_tensor(like, ub));
    Row direction = a;
    if (!upper)
        for (double &x : direction) x = -x;
    const double value = box.support_function(priv_column_tensor(like, direction)).item();
    return upper ? value : -value;
}

/// Whether the rows with one nonzero entry bound the set to a box and the others add nothing
/// (CORA's aux_isInterval): a bound of a row is its offset over the norm of the row.
bool aux_is_interval(const Polytope &P, double tol) {
    const HostH H = priv_host_h(P.constraints());
    const std::size_t n = std::size_t(H.n);
    Row lb(n, -kInf), ub(n, kInf);
    auto [rows, offsets] = priv_equality_to_inequality(H);
    std::vector<bool> check(rows.size(), false);
    for (std::size_t c = 0; c < rows.size(); ++c) {
        const Row &a = rows[c];
        if (std::all_of(a.begin(), a.end(), [](double x) { return x == 0.0; })) continue;
        // the single nonzero entry, if there is one, bounds its coordinate
        std::vector<std::size_t> nonzero;
        for (std::size_t i = 0; i < n; ++i)
            if (!within_tol(a[i], 0.0, tol)) nonzero.push_back(i);
        if (nonzero.size() != 1) {
            check[c] = true;
            continue;
        }
        const std::size_t i = nonzero[0];
        if (a[i] > 0)
            ub[i] = std::min(offsets[c] / priv_norm(a), ub[i]);
        else
            lb[i] = std::max(-offsets[c] / priv_norm(a), lb[i]);
    }
    // a row with several entries must not cut the box
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    for (std::size_t c = 0; c < rows.size(); ++c) {
        if (!check[c] || aux_box_support(like, lb, ub, rows[c], true) <= offsets[c]) continue;
        // the box is empty if it is beyond the row, else the row decides about emptiness
        if (aux_box_support(like, lb, ub, rows[c], false) > offsets[c]) return true;
        return P.representsa("emptySet", tol);
    }
    return true;
}

/// Whether the equalities and inequalities together hold a single hyperplane (CORA's
/// aux_isConHyperplane): after normalization the equalities are one.
bool aux_is_con_hyperplane(const Polytope &P, double tol) {
    HostH H = priv_host_h(P.constraints());
    if (H.Ae.size() == 1) return true;
    if (priv_compact_zeros(H, tol)) return false;
    priv_normalize_rows(H);
    priv_compact_to_equality(H, tol);
    return H.Ae.size() == 1;
}

/// Whether the set is one halfspace (CORA's aux_isHalfspace): all its rows are parallel.
bool aux_is_halfspace(const Polytope &P, double tol) {
    if (P.is_v_rep() && !P.is_h_rep()) {
        // in 1D a half-line has its infinite vertices of one sign
        if (P.dim() != 1) return false;
        bool negative = false, positive = false;
        for (const double x : P.V().data()) {
            negative = negative || x == -kInf;
            positive = positive || x == kInf;
        }
        return negative != positive;
    }
    if (P.b().numel() == 1 && P.be().numel() == 0) return true;
    if (P.be().numel() > 0) return false;
    HostH H = priv_host_h(P.constraints());
    priv_compact_zeros(H, tol);
    priv_normalize_rows(H);
    // all pairs of unit rows are equal
    bool parallel = !H.A.empty();
    for (const Row &a : H.A)
        for (const Row &c : H.A) parallel = parallel && within_tol(priv_dot(a, c), 1.0, tol);
    return parallel;
}

/// Whether the set is a single point (CORA's 'point'): one vertex in 1D, else a set without
/// interior whose affine subspace has dimension zero. The empty set is no point.
bool aux_is_point(const Polytope &P, Tol tol) {
    if (priv_polytope_is_empty(P)) return false;
    if (P.dim() == 1) return priv_polytope_vertices(P).size() == 1;
    const auto [full, subspace] = P.is_full_dim_subspace(tol);
    return !full && subspace.shape().back() == 0;
}

/// Whether the origin is the set: it contains the origin and its box is smaller than tol.
bool aux_is_origin(const Polytope &P, double tol) {
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    if (!P.contains(priv_zeros(like, P.dim(), 1), "exact", tol)) return false;
    return P.interval().norm(Tensor::Norm::two).item() <= tol;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Polytope::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "representsa");
    CORA_CHECK(type != "zonotope" && type != "parallelotope", Error::not_supported, [&] {
        return "representsa: the comparison of a polytope with '" + type + "' is not supported";
    });
    if (aux_empty_object(*this)) return aux_empty_object_answer(*this, type);
    // the types that every set answers alike
    if (type == "polytope" || type == "convexSet" || type == "levelSet") return true;
    if (type == "probZonotope") return false;
    if (type == "origin") return aux_is_origin(*this, tol);
    if (type == "point") return aux_is_point(*this, tol_arg);
    if (type == "fullspace") return priv_is_fullspace(*this, tol);
    if (type == "emptySet") return priv_polytope_is_empty(*this);
    if (type == "conHyperplane") return aux_is_con_hyperplane(*this, tol);
    if (type == "halfspace") return aux_is_halfspace(*this, tol);
    if (type == "interval") return aux_is_interval(*this, tol);
    if (type == "hyperplane")
        return is_h_rep() && A_.shape()[A_.ndim() - 2] == 0 && Ae_.shape()[0] == 1;
    if (type == "ellipsoid")
        return (dim() == 1 && is_bounded() && !priv_polytope_is_empty(*this)) ||
               aux_is_point(*this, tol_arg);
    // a capsule is a bounded segment in 1D; the other types are the bounded sets
    if (type == "capsule") return dim() == 1 && is_bounded();
    return is_bounded();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
