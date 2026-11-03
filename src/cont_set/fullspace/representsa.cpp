// representsa - whether the whole space is also a set of another type, as fullspace.representsa_
//
// Syntax:   bool res = F.representsa(type);   bool res = F.representsa(type, tol);
// Inputs:   type - CORA's name of a set type ("origin", "point", "emptySet", "fullspace",
//           "interval", "zonotope", "capsule", ...);  tol - checked and relaxes nothing; default:
//           the repository tolerance
// Outputs:  res - true for "interval", "convexSet" and "fullspace", false for every other type.
//           "levelSet" (not supported by CORA either) and "polygon" (CORA's switch has no case)
//           are Error::not_supported. In R^0 CORA's emptyObject rule answers first: false for
//           "origin", "point" and "parallelotope", true for the others. The converted set (CORA's
//           second output) is not returned: use the conversions
// See also: EmptySet::representsa, Interval::representsa
//
// CORA: contSet/@fullspace/representsa_.m
// CORA: contSet/@contSet/representsa.m

#include "cora/cont_set/fullspace/fullspace.h"

#include "cont_set/cont_set/private/priv_set_type.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::representsa(const std::string &type, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    CORA_CHECK(n_ == 0 || !priv_type_in(type, {"levelSet", "polygon"}), Error::not_supported, [&] {
        return "representsa: the comparison of the whole space to '" + type + "' is not supported";
    });
    static_cast<void>(resolve_tol(entry, tol, "representsa"));
    if (n_ == 0) return !priv_type_in(type, {"origin", "point", "parallelotope"});
    return priv_type_in(type, {"interval", "convexSet", "fullspace"});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
