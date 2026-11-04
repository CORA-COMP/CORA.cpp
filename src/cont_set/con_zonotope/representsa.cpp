// representsa - whether a constrained zonotope is also a set of another type
//
// Syntax:   bool res = cz.representsa(type);   bool res = cz.representsa(type, tol);
// Inputs:   type - CORA's name of a set type ("origin", "point", "interval", "zonotope",
//           "emptySet", "polytope", "hyperplane", ...);  tol - what counts as zero (as for a box
//           or a zonotope); default: the repository tolerance
// Outputs:  res - "emptySet": no factor vector satisfies the constraints (a linear program);
//           "origin", "point": the box around the set is that; "interval": no constraints and a
//           box; "zonotope": no constraints; "hyperplane": a point in one dimension (CORA says
//           true for every set of one dimension, docs/cora-differences.md); "conZonotope",
//           "polytope", "polyZonotope", "zonoBundle", "convexSet", "conPolyZono": true;
//           "halfspace", "probZonotope", "fullspace": false. "conHyperplane", "capsule",
//           "ellipsoid", "levelSet" and "parallelotope" are Error::not_supported, as in CORA,
//           and so is "polygon", for which CORA has no case. The converted set (CORA's second
//           output) is not returned. Not a batch
// See also: is_full_dim, Zonotope::representsa
//
// CORA: contSet/@conZonotope/representsa_.m
// CORA: contSet/@contSet/representsa.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/cont_set/private/priv_set_type.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ConZonotope::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "representsa");
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("representsa: it supports a single set, not a batch");
    });
    CORA_CHECK(type != "conHyperplane" && type != "capsule" && type != "ellipsoid" &&
                   type != "levelSet" && type != "parallelotope" && type != "polygon",
               Error::not_supported, [&] {
                   return "representsa: the comparison of a constrained zonotope with '" + type +
                          "' is not supported";
               });
    if (type == "halfspace" || type == "probZonotope" || type == "fullspace") return false;
    if (type == "emptySet") return priv_is_empty(priv_host_con_zonotope(*this));
    if (type == "origin" || type == "point") return interval().representsa(type, tol);
    if (type == "hyperplane") return dim() == 1 && interval().representsa("point", tol);
    // a set without constraints is the zonotope of its generators
    if (type == "zonotope") return A_.numel() == 0;
    if (type == "interval") return A_.numel() == 0 && Zonotope(c_, G_).representsa(type, tol);
    // conZonotope, polytope, polyZonotope, zonoBundle, convexSet and conPolyZono
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
