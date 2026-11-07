// representsa - whether a zonotope bundle is also a set of another type
//
// "origin" and "emptySet" are decided on the constrained zonotope (a linear program); "interval":
// every member is a box, or the dimension is one (CORA tests the first member in every round);
// "conZonotope", "polytope", "polyZonotope", "zonoBundle", "convexSet", "conPolyZono", "levelSet":
// true; "halfspace", "hyperplane", "probZonotope", "fullspace": false. "point", "zonotope",
// "capsule", "ellipsoid", "conHyperplane", "parallelotope" and "polygon" are Error::not_supported,
// as in CORA. The converted set is not returned. Not a batch.
//
// Syntax:   bool res = bundle.representsa(type);   bool res = bundle.representsa(type, tol);
// Inputs:   type - CORA's name of a set type
//           tol - what counts as zero; default: the repository tolerance
// Outputs:  res - whether the set is exactly such a set
// See also: ConZonotope::representsa, Zonotope::representsa
//
// CORA: contSet/@zonoBundle/representsa_.m
// CORA authors: Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_set_type.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "representsa");
    const detail::Refusal refusal = priv_batch_refusal(Z_.front().center(), "representsa");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(!priv_type_in(type, {"point", "zonotope", "capsule", "ellipsoid", "conHyperplane",
                                    "parallelotope", "polygon"}),
               Error::not_supported, [&] {
                   return "representsa: the comparison of a zonotope bundle with '" + type +
                          "' is not supported";
               });
    if (priv_type_in(type, {"halfspace", "hyperplane", "probZonotope", "fullspace"})) return false;
    if (type == "origin" || type == "emptySet") return con_zonotope().representsa(type, tol);
    if (type != "interval") return true;
    // a box has a box for every member, and so has the empty set
    if (dim() == 1 || priv_bundle_is_empty(*this)) return true;
    for (const Zonotope &Z : Z_)
        if (!Z.representsa("interval", tol)) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
