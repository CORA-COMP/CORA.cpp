// convex_hull - the convex hull of a polytope and a polytope, a box or a point, as CORA's convHull_
//
// In 1D the least and the greatest. The vertices of a halfspace polytope are found by enumeration
// (a bounded set). The hull with the empty set is the other set (CORA returns the empty set,
// docs/cora-differences.md), the hull with R^n is R^n. Not a batch.
//
// Syntax:   Pres = P.convex_hull(P2);   Pres = P.convex_hull(I);   Pres = P.convex_hull(p);
// Inputs:   P2, I - a polytope or a box of the dimension of P (bounded unless in 1D)
//           p - a point (n, 1)
// Outputs:  Pres - exact, a vertex polytope of the extreme vertices of both sets in their order
// See also: enclose, Polytope::compact
//
// CORA: contSet/@polytope/convHull_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <cmath>
#include <functional>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The second operand ------------------------------------------------------------------------------

// The second operand ------------------------------------------------------------------------------

/// The corners of the box (n, 1) as points: its sign patterns of lower and upper bounds.
Rows aux_corners(const Interval &I) {
    const int64_t n = I.dim();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    for (int64_t i = 0; i < n; ++i)
        if (std::isinf(lo[std::size_t(i)]) || std::isinf(hi[std::size_t(i)]))
            CORA_FAIL(Error::not_supported, "convex_hull: the box is unbounded");
    Rows corners;
    for (int64_t k = 0; k < (int64_t{1} << n); ++k) {
        Row corner(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            corner[std::size_t(i)] = ((k >> i) & 1) ? hi[std::size_t(i)] : lo[std::size_t(i)];
        corners.push_back(corner);
    }
    return corners;
}

/// Whether the box is R^n.
bool aux_interval_fullspace(const Interval &I) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    return !I.is_empty() && I.inf().detach().max().item() == -kInf &&
           I.sup().detach().min().item() == kInf;
}

/// The second operand of a hull: what the hull needs to know of it.
struct Operand {
    bool empty = false;
    bool fullspace = false;
    Polytope set;
    std::function<Rows()> list;     // the vertices, asked for when the hull is formed
    std::optional<Tensor> vertices; // the vertices as given, if the operand has them
};

/// The hull of the vertices of two sets in n dimensions: a vertex polytope of the extreme ones.
Polytope aux_hull(const Polytope &P, const Operand &S) {
    const int64_t n = P.dim();
    const bool p_empty = priv_polytope_is_empty(P);
    // the hull with the empty set is the other set
    if (p_empty && S.empty) return Polytope::empty(n);
    if (p_empty) return S.set;
    if (S.empty) return P;
    if (priv_is_fullspace(P) || S.fullspace) return Polytope::inf(n);
    // vertices on both sides: the hull takes its columns from them, which keeps the gradients
    if (P.is_v_rep() && S.vertices) {
        const Tensor given = Tensor::concat({P.V(), *S.vertices}, -1);
        if (n == 1) return Polytope(given);
        return Polytope(given.take(priv_extreme_points(priv_host_points(given)), -1));
    }
    Rows all = priv_polytope_vertices(P);
    const Rows second = S.list();
    all.insert(all.end(), second.begin(), second.end());
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    // in 1D the extremes, which the constructor picks; else the extreme points
    if (n == 1) return Polytope(priv_points_tensor(like, n, all));
    Rows kept;
    for (const int64_t k : priv_extreme_points(all)) kept.push_back(all[std::size_t(k)]);
    return Polytope(priv_points_tensor(like, n, kept));
}

/// Why the hull is refused, if it is: the dimensions must agree.
detail::Refusal aux_refusal(const Polytope &P, int64_t n2) {
    if (P.dim() == n2) return std::nullopt;
    return std::pair{Error::dimension_mismatch, "convex_hull: the sets differ in dimension, " +
                                                    std::to_string(P.dim()) + " and " +
                                                    std::to_string(n2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// With a polytope, a box, a point -----------------------------------------------------------------

Polytope Polytope::convex_hull(const Polytope &P2) const {
    // the vertices of a polytope are found when the hull needs them
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, P2.dim());
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    Operand S{priv_polytope_is_empty(P2), priv_is_fullspace(P2), P2,
              [&] { return priv_polytope_vertices(P2); },
              P2.is_v_rep() ? std::optional<Tensor>(P2.V()) : std::nullopt};
    return aux_hull(*this, S);
}

Polytope Polytope::convex_hull(const Interval &I) const {
    // the corners of a box are its vertices
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, I.dim());
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    Operand S{I.is_empty(), aux_interval_fullspace(I), priv_polytope_of_interval(I),
              [&] { return aux_corners(I); }, std::nullopt};
    return aux_hull(*this, S);
}

Polytope Polytope::convex_hull(const Tensor &p) const {
    detail::Entry entry;
    const bool column = p.defined() && p.ndim() == 2 && p.shape()[0] == dim() && p.shape()[1] == 1;
    CORA_CHECK(column, Error::dimension_mismatch, [&] {
        return "convex_hull: the point must be a column (" + std::to_string(dim()) + ", 1)";
    });
    // a point is the polytope of one vertex
    Operand S{false, false, Polytope(p), [&] { return priv_host_points(p); }, p};
    return aux_hull(*this, S);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
