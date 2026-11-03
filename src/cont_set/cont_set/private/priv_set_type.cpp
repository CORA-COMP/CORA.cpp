// priv_set_type - the set types of CORA's representsa, and whether a type is in a list
//
// See also: priv_set_type.h

#include "cont_set/cont_set/private/priv_set_type.h"

#include <algorithm>
#include <iterator>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_type_in(const std::string &type, std::initializer_list<const char *> types) {
    return std::any_of(types.begin(), types.end(), [&](const char *t) { return type == t; });
}

bool priv_is_set_type(const std::string &type) {
    return priv_type_in(type, {"capsule", "conPolyZono", "conHyperplane", "conZonotope",
                               "ellipsoid", "halfspace", "interval", "levelSet", "polygon",
                               "polytope", "polyZonotope", "probZonotope", "zonoBundle",
                               "zonotope", "origin", "point", "hyperplane", "parallelotope",
                               "convexSet", "emptySet", "fullspace"});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
