// is_intersecting - whether a polytope meets a set or points, as CORA's polytope.isIntersecting_
//
// "exact" decides by a linear program: two halfspace sets by the emptiness of their rows, a
// halfspace and a vertex set by a convex combination, two vertex sets likewise; a constrained
// zonotope (a zonotope and a box are one) by the least violation of the rows over its points, which
// meets if it is below 1e-6 (CORA's tolerance), a bundle by the sum of the slacks. "approx" (not
// for two polytopes, which are always exact) says false if the set lies beyond a row; it can say
// true for disjoint sets. R^n
// meets every set that is not empty. A halfspace and a halfspace meet unless they face each other
// (CORA's test reads two fields that do not exist). In one dimension the boxes decide. Not a batch.
//
// Syntax:   res = P.is_intersecting(S);   res = P.is_intersecting(S, "approx", tol);
// Inputs:   S - a polytope, a box, a zonotope, a constrained zonotope, a bundle, or points (n, N)
//           type - "exact" (default) or "approx"
//           tol - for points and halfspaces, as for contains; default: the repository tolerance
// Outputs:  res - whether the sets have a common point (all points must be in P for points)
// See also: Polytope::intersect, Polytope::contains
//
// CORA: contSet/@polytope/isIntersecting_.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
using Status = LinProgResult::Status;

// Linear programs ---------------------------------------------------------------------------------

/// Appends the row (a followed by zeros up to `width`, with a placed at `offset`) to a flat matrix.
void aux_add_row(std::vector<double> &flat, const Row &a, std::size_t offset, std::size_t width) {
    Row row(width, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) row[offset + i] = a[i];
    flat.insert(flat.end(), row.begin(), row.end());
}

/// The least violation t of the rows over the points of the constrained zonotope: min t with
/// H x - t <= d, He x = de, x = c + G xi, |xi| <= 1, A xi = b (CORA's aux_isIntersecting_P_cZ).
bool aux_meets_con_zonotope(const HostH &H, const ConZonotope &cz) {
    const HostConZonotope Z = priv_host_con_zonotope(cz);
    const std::size_t n = std::size_t(Z.n), m = std::size_t(Z.m), width = 1 + n + m;
    LinProg lp;
    lp.n = int(width);
    lp.f.assign(width, 0.0);
    lp.f[0] = 1.0;
    for (const Row &a : H.A) {
        Row row = {-1.0};
        row.insert(row.end(), a.begin(), a.end());
        row.resize(width, 0.0);
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
    }
    lp.bineq = H.b;
    // the factors stay in the unit box
    for (std::size_t j = 0; j < m; ++j) {
        for (const double sign : {-1.0, 1.0}) {
            Row row(width, 0.0);
            row[1 + n + j] = sign;
            lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
            lp.bineq.push_back(1.0);
        }
    }
    for (const Row &a : H.Ae) aux_add_row(lp.Aeq, a, 1, width);
    lp.beq = H.be;
    // x - G xi = c, and the constraints A xi = b of the set
    for (std::size_t i = 0; i < n; ++i) {
        Row row(width, 0.0);
        row[1 + i] = 1.0;
        for (std::size_t j = 0; j < m; ++j) row[1 + n + j] = -Z.G[i * m + j];
        lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
        lp.beq.push_back(Z.c[i]);
    }
    for (int64_t i = 0; i < Z.p; ++i) {
        Row row(width, 0.0);
        for (std::size_t j = 0; j < m; ++j) row[1 + n + j] = Z.A[std::size_t(i) * m + j];
        lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
        lp.beq.push_back(Z.b[std::size_t(i)]);
    }
    lp.lb.assign(width, -kInf);
    const LinProgResult r = linprog(lp);
    if (r.status == Status::infeasible) return false;
    if (r.status == Status::unbounded) return true;
    return r.fval < numerics::kPolytopeMeet;
}

/// Whether the polytope meets every zonotope of a bundle: x in P and in each one, with the slacks
/// s >= 0 of the rows summing to zero (CORA's aux_isIntersecting_P_zB; here the equalities of P
/// count too).
bool aux_meets_bundle(const HostH &H, const ZonoBundle &bundle) {
    const std::size_t n = std::size_t(H.n), p = H.A.size();
    std::size_t width = n + p;
    for (const Zonotope &Z : bundle.zonotopes())
        width += std::size_t(Z.generators().shape().back());
    LinProg lp;
    lp.n = int(width);
    lp.f.assign(width, 0.0);
    std::fill(lp.f.begin() + std::ptrdiff_t(n), lp.f.begin() + std::ptrdiff_t(n + p), 1.0);
    for (std::size_t i = 0; i < p; ++i) {
        // A x - s <= d, and the slack is nonnegative
        Row row = H.A[i];
        row.resize(width, 0.0);
        row[n + i] = -1.0;
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
        lp.bineq.push_back(H.b[i]);
        Row slack(width, 0.0);
        slack[n + i] = -1.0;
        lp.Aineq.insert(lp.Aineq.end(), slack.begin(), slack.end());
        lp.bineq.push_back(0.0);
    }
    for (const Row &a : H.Ae) aux_add_row(lp.Aeq, a, 0, width);
    lp.beq = H.be;
    std::size_t offset = n + p;
    for (const Zonotope &Z : bundle.zonotopes()) {
        // x - G xi = c with |xi| <= 1 for every zonotope
        const std::vector<double> G = Z.generators().data(), c = Z.center().data();
        const std::size_t m = std::size_t(Z.generators().shape().back());
        for (std::size_t i = 0; i < n; ++i) {
            Row row(width, 0.0);
            row[i] = 1.0;
            for (std::size_t j = 0; j < m; ++j) row[offset + j] = -G[i * m + j];
            lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
            lp.beq.push_back(c[i]);
        }
        // the factors of the zonotope lie in the box [-1, 1]
        for (std::size_t j = 0; j < m; ++j) {
            for (const double sign : {-1.0, 1.0}) {
                Row row(width, 0.0);
                row[offset + j] = sign;
                lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
                lp.bineq.push_back(1.0);
            }
        }
        offset += m;
    }
    lp.lb.assign(width, -kInf);
    const LinProgResult r = linprog(lp);
    return r.status != Status::infeasible &&
           !(r.status == Status::optimal && r.fval > 0.0 && !within_tol(r.fval, 0.0));
}

/// Whether a point in the convex hull of the vertices V (n, p) satisfies the rows of H (CORA's
/// aux_isIntersecting_Hpoly_Vpoly): the weights are nonnegative and sum to one.
bool aux_meets_vertices(const HostH &H, const Rows &V) {
    const std::size_t n = std::size_t(H.n), p = V.size();
    LinProg lp;
    lp.n = int(n + p);
    lp.f.assign(n + p, 0.0);
    for (const Row &a : H.A) aux_add_row(lp.Aineq, a, 0, n + p);
    lp.bineq = H.b;
    for (std::size_t k = 0; k < p; ++k) {
        Row row(n + p, 0.0);
        row[n + k] = -1.0;
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
        lp.bineq.push_back(0.0);
    }
    for (const Row &a : H.Ae) aux_add_row(lp.Aeq, a, 0, n + p);
    lp.beq = H.be;
    // x = sum of weighted vertices, and the weights sum to one
    for (std::size_t i = 0; i < n; ++i) {
        Row row(n + p, 0.0);
        row[i] = -1.0;
        for (std::size_t k = 0; k < p; ++k) row[n + k] = V[k][i];
        lp.Aeq.insert(lp.Aeq.end(), row.begin(), row.end());
        lp.beq.push_back(0.0);
    }
    Row ones(n + p, 0.0);
    std::fill(ones.begin() + std::ptrdiff_t(n), ones.end(), 1.0);
    lp.Aeq.insert(lp.Aeq.end(), ones.begin(), ones.end());
    lp.beq.push_back(1.0);
    lp.lb.assign(n + p, -kInf);
    return linprog(lp).status != Status::infeasible;
}

// The tests ---------------------------------------------------------------------------------------

/// The least value of a'x over the set (the support function of -a, negated).
double aux_lowest(const ContSet &S, const Row &a, const Tensor &like) {
    Row negated = a;
    for (double &x : negated) x = -x;
    return -S.support_function(priv_column_tensor(like, negated)).item();
}

/// Whether two polytopes meet, by the forms they hold.
bool aux_meets_polytope(const Polytope &P, const Polytope &S) {
    if (priv_is_fullspace(P) || priv_is_fullspace(S)) return true;
    if (P.is_v_rep() && S.is_v_rep())
        return aux_meets_vertices(priv_host_h(S.constraints()), priv_host_points(P.V()));
    if (P.is_h_rep() && S.is_h_rep()) {
        const Polytope::Constraints C1 = P.constraints(), C2 = S.constraints();
        HostH both = priv_host_h(C1);
        const HostH other = priv_host_h(C2);
        both.A.insert(both.A.end(), other.A.begin(), other.A.end());
        both.b.insert(both.b.end(), other.b.begin(), other.b.end());
        both.Ae.insert(both.Ae.end(), other.Ae.begin(), other.Ae.end());
        both.be.insert(both.be.end(), other.be.begin(), other.be.end());
        return !priv_is_empty(both);
    }
    // a halfspace set and a vertex set
    const Polytope &H = P.is_h_rep() && !P.is_v_rep() ? P : S;
    const Polytope &V = &H == &P ? S : P;
    return aux_meets_vertices(priv_host_h(H.constraints()), priv_host_points(V.V()));
}

/// The test of "approx": whether S lies beyond a row of P (false) or not (true).
bool aux_not_beyond(const Polytope &P, const ContSet &S, double tol) {
    const Polytope::Constraints C = P.constraints();
    const HostH H = priv_host_h(C);
    for (std::size_t i = 0; i < H.A.size(); ++i)
        if (H.b[i] < aux_lowest(S, H.A[i], C.A)) return false;
    // an equality row needs its offset inside the range of S
    for (std::size_t i = 0; i < H.Ae.size(); ++i) {
        const double low = aux_lowest(S, H.Ae[i], C.A);
        const double high = S.support_function(priv_column_tensor(C.A, H.Ae[i])).item();
        if (!(low <= H.be[i] && H.be[i] <= high) && !within_tol(H.be[i], low, tol) &&
            !within_tol(H.be[i], high, tol))
            return false;
    }
    return true;
}

/// Whether a halfspace meets the set: the set's lowest value along the normal is within the offset.
bool aux_halfspace_meets(const Polytope &P, const ContSet &S, double tol) {
    const Polytope::Constraints C = P.constraints();
    const HostH H = priv_host_h(C);
    const double bound = aux_lowest(S, H.A[0], C.A);
    const double offset = *std::min_element(H.b.begin(), H.b.end());
    return bound <= offset || within_tol(bound, offset, tol);
}

/// The answer for a set S that is not a polytope and not points, after the checks.
template <class Convert>
bool aux_meets_set(const Polytope &P, const ContSet &S, const std::string &type, double tol,
                   Convert convert) {
    // R^n meets every set that is not empty
    if (priv_is_fullspace(P)) return !S.interval().is_empty();
    if (P.dim() == 1) return P.interval().is_intersecting(S.interval(), tol);
    if (P.representsa("halfspace", tol)) return aux_halfspace_meets(P, S, tol);
    if (type == "approx") return aux_not_beyond(P, S, tol);
    return convert(priv_host_h(P.constraints()));
}

/// Why a call is refused: the type, a batch, a dimension.
detail::Refusal aux_refusal(const Polytope &P, const ContSet &S, const std::string &type) {
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    if (const Refusal r = priv_intersecting_refusal(type, like, like)) return r;
    if (like.ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("a linear program per set: single sets")};
    if (S.dim() != P.dim())
        return std::pair{Error::dimension_mismatch, "the sets differ in dimension, " +
                                                        std::to_string(P.dim()) + " and " +
                                                        std::to_string(S.dim())};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Polytopes ---------------------------------------------------------------------------------------

bool Polytope::is_intersecting(const Polytope &S, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol =
        resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "is_intersecting");
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    // the whole space meets every set that is not empty
    if (priv_is_fullspace(*this)) return !S.representsa("emptySet", machine_eps());
    if (priv_is_fullspace(S)) return !representsa("emptySet", machine_eps());
    if (dim() == 1) return interval().is_intersecting(S.interval(), tol);
    if (representsa("halfspace", tol)) return aux_halfspace_meets(*this, S, tol);
    return aux_meets_polytope(*this, S);
}

// Boxes, zonotopes, constrained zonotopes and bundles ---------------------------------------------

bool Polytope::is_intersecting(const ConZonotope &S, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    // the tolerance is checked; the exact test needs none
    static_cast<void>(
        resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "is_intersecting"));
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    // the bounds of "approx" are support functions, which a constrained zonotope does not have yet
    CORA_CHECK(type == "exact", Error::not_supported, [] {
        return std::string("is_intersecting: 'approx' needs the support function of a constrained "
                           "zonotope, which is not ported");
    });
    // R^n meets every set that is not empty; else the program over both
    if (priv_is_fullspace(*this)) return !S.interval().is_empty();
    return aux_meets_con_zonotope(priv_host_h(constraints()), S);
}

bool Polytope::is_intersecting(const Zonotope &Z, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol =
        resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "is_intersecting");
    const detail::Refusal refusal = aux_refusal(*this, Z, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets_set(*this, Z, type, tol,
                         [&](const HostH &H) { return aux_meets_con_zonotope(H, ConZonotope(Z)); });
}

bool Polytope::is_intersecting(const Interval &I, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol =
        resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "is_intersecting");
    CORA_CHECK(I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "is_intersecting: the interval must be a box with column bounds, got the shape " +
               shape_text(I.inf().shape());
    });
    // an empty box meets nothing
    if (I.is_empty()) return false;
    const detail::Refusal refusal = aux_refusal(*this, I, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets_set(*this, I, type, tol,
                         [&](const HostH &H) { return aux_meets_con_zonotope(H, ConZonotope(I)); });
}

bool Polytope::is_intersecting(const ZonoBundle &S, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol =
        resolve_tol(entry, tol_arg, (is_h_rep() ? A_ : V_).dtype(), "is_intersecting");
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets_set(*this, S, type, tol,
                         [&](const HostH &H) { return aux_meets_bundle(H, S); });
}

// Points ------------------------------------------------------------------------------------------

bool Polytope::is_intersecting(const Tensor &p, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "is_intersecting");
    // the refusal reads the dimension from the rows of a center column
    const Tensor center = priv_column_tensor(like, Row(static_cast<std::size_t>(dim()), 0.0));
    const Refusal refusal = priv_intersecting_points_refusal(type, center, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    // points are in the set or not: all of them
    return contains(p, type, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
