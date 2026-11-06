// minkowski_difference - the Minkowski difference of a constrained zonotope and a set
//
// "exact" intersects the translates cz - v by the corners v of S (CORA's vertices; at most 12
// generators); "inner:vertices" translates by the center of S and intersects cz + g and cz - g
// for each generator g, a subset of the difference; "inner:Vinod" scales the factors by one minus
// the support of S along the directions the constraints give them (Vinod et al.), a subset, and
// the empty set if that exceeds 1; the inner ones take a zonotope or a box. A point translates
// the set. Not a batch.
//
// Syntax:   cZres = cz.minkowski_difference(S);   cZres = cz.minkowski_difference(S, type);
//           cZres = cz.minkowski_difference(p);
// Inputs:   S - a zonotope or a box, or a point p (n, 1)
//           type - "exact" (default), "inner:vertices" or "inner:Vinod"
// Outputs:  cZres - {x | x + S in cz}
// See also: plus, intersect, Zonotope::mink_diff
//
// CORA: contSet/@conZonotope/minkDiff.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Types -------------------------------------------------------------------------------------------

/// The columns of V (n, K) once each, in order, as (n, 1) tensors.
std::vector<Tensor> aux_distinct_columns(const Tensor &V) {
    const int64_t n = V.shape()[0], K = V.shape()[1];
    const std::vector<double> v = V.data();
    std::set<std::vector<double>> seen;
    std::vector<Tensor> columns;
    for (int64_t k = 0; k < K; ++k) {
        std::vector<double> column(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) column[i] = v[i * K + k];
        if (seen.insert(column).second) columns.push_back(V.slice(-1, k, k + 1));
    }
    return columns;
}

/// Intersect the translates cz - v of every vertex v of S (CORA's 'exact').
ConZonotope aux_exact(const ConZonotope &cz, const Zonotope &S) {
    // the corners of S are its vertices (CORA lists the vertices; the same translates result)
    const std::vector<Tensor> vertices = aux_distinct_columns(priv_corners(S));
    ConZonotope result = cz.plus(vertices[0] * -1.0);
    for (std::size_t i = 1; i < vertices.size(); ++i)
        result = result.intersect(cz.plus(vertices[i] * -1.0));
    return result;
}

/// Translate by the center of S and, for each generator g, intersect cz + g with cz - g.
ConZonotope aux_inner_vertices(const ConZonotope &cz, const Zonotope &S) {
    ConZonotope result = cz.plus(S.center() * -1.0);
    const Tensor &G = S.generators();
    for (int64_t i = 0; i < G.shape().back(); ++i) {
        const Tensor g = G.slice(-1, i, i + 1);
        result = result.plus(g).intersect(result.plus(g * -1.0));
    }
    return result;
}

/// The scaling of the factors by Vinod et al.: one minus the support of S0 = S - center along the
/// direction each factor has in the pseudoinverse of [G; A].
ConZonotope aux_inner_vinod(const ConZonotope &cz, const Zonotope &S) {
    const int64_t n = cz.dim(), p = cz.A().shape()[0], N = cz.A().shape()[1];
    const Tensor lifted = Tensor::concat({cz.generators(), cz.A()}, -2);
    const Tensor pick = Tensor::concat({cz.c().eye_like(n), priv_zeros(cz.c(), p, n)}, -2);
    const Tensor Gamma = lifted.pinv().matmul(pick); // (N, n)
    // the support of the centered S along row i of Gamma: sum_k |Gamma_i . g_k|
    const Tensor support = Gamma.matmul(S.generators()).abs().sum(-1, true); // (N, 1)
    const std::vector<double> s = support.data();
    for (const double v : s)
        if (v > 1.0) return ConZonotope::empty(n);
    const Tensor D = (support.ones_like() - support).reshape({1, N});
    return {cz.c() - S.center(), cz.generators().multiply(D), cz.A().multiply(D), cz.b()};
}

// The dispatch ------------------------------------------------------------------------------------

/// The difference of cz and the zonotope S by the type.
ConZonotope aux_difference(const ConZonotope &cz, const Zonotope &S, const std::string &type) {
    // the generic entry has checked the dimension; the rest is this class's own
    if (type != "exact" && type != "inner:vertices" && type != "inner:Vinod")
        CORA_FAIL(Error::wrong_value, "minkowski_difference: the type must be exact, "
                                      "inner:vertices or inner:Vinod, got '" +
                                          type + "'");
    if (cz.c().ndim() != 2 || S.center().ndim() != 2)
        CORA_FAIL(Error::not_supported,
                  "minkowski_difference: it supports single sets, not a batch");
    if (type == "inner:vertices") return aux_inner_vertices(cz, S);
    if (type == "inner:Vinod") return aux_inner_vinod(cz, S);
    const Zonotope compacted = S.compact();
    if (compacted.generators().shape().back() > kMaxCornerGenerators)
        CORA_FAIL(Error::not_supported,
                  "minkowski_difference: a set of more than 12 generators has too many vertices "
                  "to intersect one by one");
    return aux_exact(cz, compacted);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::minkowski_difference(const Tensor &p) const { return plus(p * -1.0); }

Set ConZonotope::minkowski_difference_impl(const ContSet &S2,
                                           const std::optional<std::string> &type) const {
    const std::string method = type.value_or("exact");
    if (S2.class_id() == ClassId::zonotope)
        return aux_difference(*this, static_cast<const Zonotope &>(S2), method);
    if (S2.class_id() == ClassId::interval)
        return aux_difference(*this, static_cast<const Interval &>(S2).zonotope(), method);
    CORA_FAIL(Error::noops,
              "minkowski_difference is not implemented for ConZonotope and " + S2.class_name());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
