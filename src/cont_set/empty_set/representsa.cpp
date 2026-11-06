// representsa - whether the empty set is also a set of another type, as emptySet.representsa_
//
// Syntax:   bool res = E.representsa(type);   bool res = E.representsa(type, tol);
// Inputs:   type - CORA's name of a set type ("origin", "point", "emptySet", "fullspace",
//           "interval", "zonotope", "capsule", ...);  tol - checked and relaxes nothing; default:
//           the repository tolerance
// Outputs:  res - true for every type that has an empty member, and for "convexSet"; false for
//           "origin", "point", "halfspace", "hyperplane", "parallelotope", "probZonotope" and
//           "fullspace". "levelSet" (not supported by CORA either) and "polygon" (CORA's switch
//           has no case) are Error::not_supported. The converted set (CORA's second output) is
//           not returned: use the conversions
// See also: Fullspace::representsa, Interval::representsa
//
// CORA: contSet/@emptySet/representsa_.m
// CORA: contSet/@contSet/representsa.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

#include "cont_set/cont_set/private/priv_set_type.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::representsa(const std::string &type, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    CORA_CHECK(!priv_type_in(type, {"levelSet", "polygon"}), Error::not_supported, [&] {
        return "representsa: the comparison of the empty set to '" + type + "' is not supported";
    });
    static_cast<void>(resolve_tol(entry, tol, "representsa"));
    return !priv_type_in(type, {"origin", "point", "halfspace", "hyperplane", "parallelotope",
                                "probZonotope", "fullspace"});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
