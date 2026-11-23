// representsa - whether a polynomial zonotope is also a set of another type, as CORA's representsa_
//
// Judged on the compact set, except "point", which is CORALean's test on the set as it stands
// (every generator entry zero; "emptySet" is false, a polynomial zonotope is never empty).
// Decided here: the types that every polynomial zonotope is, the ones
// none is, "origin", "point", "capsule", "ellipsoid", and "zonotope", "interval" and
// "parallelotope" through the exponents (each factor in one generator). "conZonotope",
// "polytope" and "zonoBundle" are answered for dimension 1 and for a zonotope; any other set
// would need CORA's vertex and cone test (convhulln, linear programs) and is Error::not_supported.
// The converted set (CORA's second output) is not returned.
//
// Syntax:   res = P.representsa(type);   res = P.representsa(type, tol);
// Inputs:   type - CORA's name of a set type
//           tol - what counts as zero; default: the repository tolerance
// Outputs:  res - whether the set is exactly such a set
// See also: is_full_dim, compact
//
// CORA: contSet/@polyZonotope/representsa_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The types of CORA's representsa.
const char *const kTypes[] = {
    "capsule",    "conPolyZono", "conHyperplane", "conZonotope", "ellipsoid",    "halfspace",
    "interval",   "levelSet",    "polygon",       "polytope",    "polyZonotope", "probZonotope",
    "zonoBundle", "zonotope",    "origin",        "point",       "hyperplane",   "parallelotope",
    "convexSet",  "emptySet",    "fullspace"};

/// CORA's aux_isZonotope: the merged exponents have one factor per generator, none shared.
bool aux_is_zonotope(const PolyZonotope &P) {
    if (P.num_dependent() == 0) return true;
    if (P.exponents().size() != P.ids().size()) return false;
    std::vector<int> per_factor(P.ids().size(), 0);
    for (const std::vector<int> &col : P.exponents()) {
        int used = 0;
        for (std::size_t i = 0; i < col.size(); ++i)
            if (col[i] != 0) {
                ++used;
                ++per_factor[i];
            }
        if (used > 1) return false;
    }
    return std::all_of(per_factor.begin(), per_factor.end(), [](int k) { return k <= 1; });
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(std::find(std::begin(kTypes), std::end(kTypes), type) != std::end(kTypes),
               Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "representsa");
    CORA_CHECK(c_.ndim() == 1, Error::not_supported,
               [] { return std::string("representsa: it supports a single set, not a batch"); });
    const PolyZonotope P = compact();
    const int64_t n = P.dim();
    const bool point = P.num_dependent() == 0 && P.independent_generators().shape()[0] == 0;
    if (type == "origin")
        return (point && P.center().abs().max().item() <= tol) || P.interval().norm().item() <= tol;
    if (type == "point") {
        // CORALean's test on the set as it is: every generator entry zero up to tol
        const auto zero = [&](const Tensor &g) {
            return g.shape()[0] == 0 || g.abs().max().item() <= tol;
        };
        return zero(G_) && zero(GI_);
    }
    // the types decided by the shape of the compact set
    if (type == "capsule" || type == "ellipsoid") return n == 1 || point;
    if (type == "conPolyZono" || type == "polyZonotope") return true;
    if (type == "zonotope") return n == 1 || aux_is_zonotope(P);
    if (type == "interval")
        return n == 1 || (aux_is_zonotope(P) && P.zonotope().representsa("interval", tol));
    if (type == "parallelotope")
        return n == 1 || (P.num_dependent() == 0 && P.zonotope().representsa(type, tol));
    if (type == "conZonotope" || type == "polytope" || type == "zonoBundle") {
        CORA_CHECK(n == 1 || aux_is_zonotope(P), Error::not_supported, [&] {
            return "representsa: the test of a polynomial zonotope for '" + type +
                   "' (CORA's vertex and cone test) is not ported";
        });
        return true;
    }
    CORA_CHECK(type != "conHyperplane" && type != "levelSet" && type != "polygon" &&
                   type != "convexSet",
               Error::not_supported, [&] {
                   return "representsa: the comparison of a polynomial zonotope with '" + type +
                          "' is not supported";
               });
    // halfspace, probZonotope, hyperplane, emptySet, fullspace
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
