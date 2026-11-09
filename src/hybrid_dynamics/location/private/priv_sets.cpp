// priv_sets - the set helpers of the location's private functions
//
// A reachable set of a location is a zonotope or a zonotope bundle; these take it apart, put it
// together, test it against a polytope as CORA's isIntersecting_(P, S, 'approx', 1e-8), and give
// the bases of a guard hyperplane.
//
// Syntax:   std::vector<Zonotope> parts = priv_zonotopes(S);   Set S = priv_bundle(parts);
// See also: priv.h
//
// CORA.cpp only: CORA's sets carry these operations themselves (zonoBundle.Z, isIntersecting_).
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Zonotope> priv_zonotopes(const Set &S) {
    if (S.is<Zonotope>()) return {S.as<Zonotope>()};
    return S.as<ZonoBundle>().zonotopes();
}

Set priv_bundle(std::vector<Zonotope> parts) {
    if (parts.size() == 1) return std::move(parts.front());
    return ZonoBundle(std::move(parts));
}

bool priv_meets(const Polytope &P, const Set &S) {
    // CORA's isIntersecting_(P, S, 'approx', 1e-8) for the two classes a location holds
    if (S.is<Zonotope>())
        return P.is_intersecting(S.as<Zonotope>(), "approx", numerics::kGuardMeet);
    return P.is_intersecting(S.as<ZonoBundle>(), "approx", numerics::kGuardMeet);
}

Tensor priv_basis_of(const Tensor &a) {
    const Rows Q = priv_gram_schmidt(a.data());
    const std::size_t n = Q.size();
    // the vectors are the columns
    std::vector<double> B(n * n);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) B[i * n + j] = Q[j][i];
    return Tensor::like(a, B, {int64_t(n), int64_t(n)});
}

std::pair<Tensor, double> priv_hyperplane(const Polytope &guard) {
    // CORA's normalizeConstraints(guard, 'A'): the equality as a unit normal and its offset
    const Tensor a = guard.Ae().slice(-2, 0, 1).matrix_transpose();
    const double norm = std::sqrt(a.square().sum().item());
    return {a / norm, guard.be().data()[0] / norm};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
