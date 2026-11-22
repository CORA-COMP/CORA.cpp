// contains - whether an ellipsoid contains points, a zonotope or a box, as CORA's contains_
//
// The default tol is CORA's contains default. For a set: an unbounded set is not contained, an
// empty one is; a set that is a point as a point; else the zonotope (a box as its zonotope) is
// contained iff all its vertices are, found by a search over the signs of the generators, longest
// first, that cuts a branch when no completion can exceed 1 + tol. An ellipsoid that is a point
// contains only points within tol of it, an empty one only the empty set. A set that is a point
// is compared with the tolerance (CORA compares exactly, and passes it to a function that takes
// fewer arguments, docs/cora-differences.md). Error::not_supported: "approx" (an SDP) and an
// ellipsoid as the contained set unless it is a point or empty. Not a batch.
//
// Syntax:   Mask m = E.contains(p);   bool res = E.contains(Z, method, tol);   E.contains(I);
// Inputs:   p - points (..., n), one per row; the mask has one flag per point
//           Z, I - a zonotope, a box
//           S - an ellipsoid
//           method - "exact" (default; the points ignore it)
//           tol - allowed excess of the ellipsoid norm over 1; default: the repository tolerance
// Outputs:  res - for points, the ellipsoid norm of p - q is at most 1 + tol
// See also: ellipsoid_norm, Zonotope::contains
// References:
//    [1] E. A. Yildirim. "On the minimum volume covering ellipsoid of ellipsoids", SIAM Journal on
//        Optimization 17(3), 2006, pp. 621-641.
//    [2] A. Kulmburg, L. Schaefer, M. Althoff. "Approximability of the containment problem for
//        zonotopes and ellipsotopes", IEEE TAC, 2025.
//
// CORA: contSet/@ellipsoid/contains_.m
// CORA authors: Victor Gassmann, Niklas Kochdumper, Adrian Kulmburg, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Search ------------------------------------------------------------------------------------------

// The search keeps CORA's layout: points and generators are the columns of P and G (n, N).

/// The center of E as a column (n, 1).
Tensor aux_center(const Ellipsoid &E) { return E.q().expand_dims(-1); }

/// The ellipsoid norm of every column of P (n, N) after the center q is subtracted, on the host.
std::vector<double> aux_norms(const Ellipsoid &E, const Tensor &P) {
    std::vector<double> norms;
    norms.reserve(std::size_t(P.shape()[1]));
    for (int64_t j = 0; j < P.shape()[1]; ++j)
        norms.push_back(priv_ellipsoid_norm(E, P.slice(1, j, j + 1) - aux_center(E)).item());
    return norms;
}

/// Whether every column of P (n, N) lies in the nondegenerate or degenerate ellipsoid E, up to tol.
bool aux_contains_points(const Ellipsoid &E, const Tensor &P, double tol) {
    const std::vector<double> norms = aux_norms(E, P);
    return std::all_of(norms.begin(), norms.end(), [&](double x) { return x <= 1.0 + tol; });
}

/// Whether the zonotope with the generators G (n, m) lies in the ball of the ellipsoid norm of E,
/// up to tol: a search over the signs of the generators, longest first.
bool aux_inside(const Ellipsoid &E, const Tensor &G, const std::vector<double> &norms,
                const std::vector<int64_t> &order, std::size_t next, const Tensor &sum, double rest,
                double tol) {
    const double value = priv_ellipsoid_norm(E, sum).item();
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
    const Tensor all = Tensor::concat({c - aux_center(E), G}, 1);
    std::vector<double> norms = aux_norms(E, all);
    std::vector<int64_t> order(norms.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return norms[a] > norms[b]; });
    const double total = std::accumulate(norms.begin(), norms.end(), 0.0);
    return aux_inside(E, all, norms, order, 0, all.zeros_like({E.dim(), 1}), total, tol);
}

// Refusals ----------------------------------------------------------------------------------------

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
/// (a point `q`, a column, empty, or neither) is contained.
std::optional<bool> aux_degenerate(const Ellipsoid &E, bool s_empty, const Tensor *s_point,
                                   double tol) {
    if (priv_is_empty(E)) return s_empty;
    if (!priv_is_point(E)) return std::nullopt;
    if (s_point) return (*s_point - aux_center(E)).abs().max().item() <= tol;
    return s_empty;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Entries -----------------------------------------------------------------------------------------

Mask Ellipsoid::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "contains");
    CORA_CHECK(p.defined() && p.ndim() >= 1, Error::wrong_value, [] {
        return std::string("contains: the points must be (..., n), one point per row");
    });
    const detail::Refusal refusal = aux_refusal(*this, p.size(-1), method, false);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // one flag per point; an empty ellipsoid contains none, and no points violate nothing
    Shape batch = p.shape();
    batch.pop_back();
    const int64_t n = dim(), N = n == 0 ? 0 : p.numel() / n;
    std::vector<double> flags(std::size_t(std::max<int64_t>(N, 0)), 0.0);
    if (N > 0 && !priv_is_empty(*this)) {
        // a point ellipsoid holds the points within tol of it, entry by entry
        if (priv_is_point(*this)) {
            const std::vector<double> x = p.data(), c = q_.data();
            for (int64_t k = 0; k < N; ++k) {
                double far = 0.0;
                for (int64_t i = 0; i < n; ++i) far = std::max(far, std::abs(x[k * n + i] - c[i]));
                flags[k] = far <= tol ? 1.0 : 0.0;
            }
        } else {
            // else the ellipsoid norm of each point at most 1 + tol
            const std::vector<double> norms =
                aux_norms(*this, p.reshape({N, n}).matrix_transpose());
            for (int64_t k = 0; k < N; ++k) flags[k] = norms[k] <= 1.0 + tol ? 1.0 : 0.0;
        }
    }
    return Mask(Tensor::like(Q_, flags, batch));
}

bool Ellipsoid::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "contains");
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // a zonotope that is a point is decided as a point
    const Tensor &c = S.center().expand_dims(-1);
    const bool s_point = S.representsa("point");
    if (const auto done = aux_degenerate(*this, false, s_point ? &c : nullptr, tol)) return *done;
    if (s_point) return aux_contains_points(*this, c, tol);
    return aux_contains_zonotope(*this, c, S.generators().matrix_transpose(), tol);
}

bool Ellipsoid::contains(const Interval &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol_arg, Q_.dtype(), "contains"));
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // an empty box is contained anywhere, an unbounded one nowhere (E is bounded)
    if (S.is_empty()) return true;
    if (!S.is_bounded().all()) return false;
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
    const Tensor c = aux_center(S);
    if (const auto done = aux_degenerate(*this, priv_is_empty(S), s_point ? &c : nullptr, tol))
        return *done;
    if (priv_is_empty(S)) return true;
    if (s_point) return aux_contains_points(*this, c, tol);
    // an ellipsoid in an ellipsoid needs an SDP
    CORA_FAIL(
        Error::not_supported,
        "contains: an ellipsoid in an ellipsoid is decided by an SDP, which is not available");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
