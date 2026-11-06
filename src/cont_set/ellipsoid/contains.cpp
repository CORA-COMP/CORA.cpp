// contains - whether an ellipsoid contains points, a zonotope or a box, as CORA's contains_
//
// Syntax:   bool res = E.contains(p);   bool res = E.contains(Z, method, tol);   E.contains(I);
// Inputs:   p - points as columns (n, N), all must be inside;  Z, I - a zonotope, a box;
//           S - an ellipsoid;  method - "exact" (default; for points the method is not used);
//           tol - allowed excess of the ellipsoid norm over 1; default: the repository tolerance
//           (docs/architecture.md; CORA's contains default). Not a batch
// Outputs:  res - for points, the ellipsoid norm of p - q is at most 1 + tol. For a set: an
//           unbounded set is not contained, an empty one is; a set that is a point as a point;
//           else the zonotope (a box as its zonotope) is contained iff all its vertices are,
//           found by a search over the signs of the generators, longest first, that cuts a branch
//           when no completion can exceed 1 + tol. An ellipsoid that is a point contains only
//           points within tol of it, an empty one only the empty set. A set that is a point is
//           compared with the tolerance (CORA compares exactly, and passes it to a function that
//           takes fewer arguments, docs/cora-differences.md). Error::not_supported: "approx" (an
//           SDP) and an ellipsoid as the contained set unless it is a point or empty
// See also: ellipsoid_norm, Zonotope::contains
//
// CORA: contSet/@ellipsoid/contains_.m
// CORA authors: Victor Gassmann, Niklas Kochdumper, Adrian Kulmburg, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Search ------------------------------------------------------------------------------------------

/// The ellipsoid norm of every column of P (n, N) after the center q is subtracted, on the host.
std::vector<double> aux_norms(const Ellipsoid &E, const Tensor &P) {
    std::vector<double> norms;
    norms.reserve(std::size_t(P.shape()[1]));
    for (int64_t j = 0; j < P.shape()[1]; ++j)
        norms.push_back(E.ellipsoid_norm(P.slice(1, j, j + 1) - E.q()).item());
    return norms;
}

/// Whether every column of P (n, N) lies in the nondegenerate or degenerate ellipsoid E, up to tol.
bool aux_contains_points(const Ellipsoid &E, const Tensor &P, double tol) {
    const std::vector<double> norms = aux_norms(E, P);
    return std::all_of(norms.begin(), norms.end(), [&](double x) { return x <= 1.0 + tol; });
}

/// Whether the zonotope with the generators G (n, m) (columns), of the center offset first, lies in
/// the ball of the ellipsoid norm of E, up to tol: no point G nu, |nu|_inf <= 1, is farther out.
/// A search over the signs of the generators, longest first; `rest` is the sum of the norms
/// of the generators not yet assigned, the most the partial sum can still grow.
bool aux_inside(const Ellipsoid &E, const Tensor &G, const std::vector<double> &norms,
                const std::vector<int64_t> &order, std::size_t next, const Tensor &sum, double rest,
                double tol) {
    const double value = E.ellipsoid_norm(sum).item();
    // a partial sum is a point of the set
    if (value > 1.0 + tol) return false;
    if (next == order.size() || value + rest <= 1.0 + tol) return true;
    const int64_t j = order[next];
    const double remaining = rest - norms[j];
    const Tensor column = G.slice(1, j, j + 1);
    return aux_inside(E, G, norms, order, next + 1, sum + column, remaining, tol) &&
           aux_inside(E, G, norms, order, next + 1, sum - column, remaining, tol);
}

/// Whether the zonotope c + G [-1, 1]^m lies in E up to tol (the offset c - q is a generator with
/// a factor of +-1, which the symmetry of the norm makes equivalent).
bool aux_contains_zonotope(const Ellipsoid &E, const Tensor &c, const Tensor &G, double tol) {
    const Tensor all = Tensor::concat({c - E.q(), G}, 1);
    std::vector<double> norms = aux_norms(E, all);
    std::vector<int64_t> order(norms.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return norms[a] > norms[b]; });
    const double total = std::accumulate(norms.begin(), norms.end(), 0.0);
    return aux_inside(E, all, norms, order, 0, all.zeros_like({E.dim(), 1}), total, tol);
}

/// Why the call is refused, if it is: a batch, a method, or the wrong dimension. Only points
/// are decided with "approx" (exactly, as CORA does); for a set it needs an SDP.
detail::Refusal aux_refusal(const Ellipsoid &E, int64_t n, const std::string &method, bool is_set) {
    if (E.Q().ndim() != 2)
        return std::pair{Error::not_supported, std::string("contains: it decides one ellipsoid; "
                                                           "a batch is not supported")};
    if (method != "exact" && method != "approx")
        return std::pair{Error::wrong_value,
                         "contains: the method must be 'exact' or 'approx', got '" + method + "'"};
    if (is_set && method == "approx")
        return std::pair{Error::not_supported,
                         std::string("contains: the method 'approx' needs an SDP solver")};
    if (n != E.dim())
        return std::pair{Error::dimension_mismatch, "contains: the set has dimension " +
                                                        std::to_string(n) + ", the ellipsoid " +
                                                        std::to_string(E.dim())};
    return std::nullopt;
}

/// The outcome for an ellipsoid that is a point p0 or empty, if it is one: whether the set
/// (a point `q`, empty, or neither) is contained.
std::optional<bool> aux_degenerate(const Ellipsoid &E, bool s_empty, const Tensor *s_point,
                                   double tol) {
    if (priv_is_empty(E)) return s_empty;
    if (!priv_is_point(E)) return std::nullopt;
    if (s_point) return (*s_point - E.q()).abs().max().item() <= tol;
    return s_empty;
}

} // namespace


// ===========================================  MAIN  =========================================== //


// Entries -----------------------------------------------------------------------------------------

bool Ellipsoid::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "contains");
    CORA_CHECK(p.defined() && p.ndim() == 2, Error::wrong_value, [] {
        return std::string("contains: the points must be a matrix (n, N), one point per column");
    });
    const detail::Refusal refusal = aux_refusal(*this, p.shape()[0], method, false);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // no points: nothing to violate
    if (p.numel() == 0) return true;
    if (priv_is_empty(*this)) return false;
    if (priv_is_point(*this)) return (p - q_).abs().max().item() <= tol;
    return aux_contains_points(*this, p, tol);
}

bool Ellipsoid::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "contains");
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // a zonotope that is a point is decided as a point
    const Tensor &c = S.center();
    const bool s_point = S.representsa("point");
    if (const auto done = aux_degenerate(*this, false, s_point ? &c : nullptr, tol)) return *done;
    if (s_point) return aux_contains_points(*this, c, tol);
    return aux_contains_zonotope(*this, c, S.generators(), tol);
}

bool Ellipsoid::contains(const Interval &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol_arg, Q_.dtype(), "contains"));
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // an empty box is contained anywhere, an unbounded one nowhere (E is bounded)
    if (S.is_empty()) return true;
    if (!S.is_bounded()) return false;
    return contains(S.zonotope(), method, tol_arg);
}

bool Ellipsoid::contains(const Ellipsoid &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "contains");
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(S.Q().ndim() == 2, Error::not_supported, [] {
        return std::string("contains: it decides one ellipsoid; a batch is not supported");
    });
    const bool s_point = priv_is_point(S);
    if (const auto done = aux_degenerate(*this, priv_is_empty(S), s_point ? &S.q() : nullptr, tol))
        return *done;
    if (priv_is_empty(S)) return true;
    if (s_point) return aux_contains_points(*this, S.q(), tol);
    CORA_FAIL(
        Error::not_supported,
        "contains: an ellipsoid in an ellipsoid is decided by an SDP, which is not available");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
