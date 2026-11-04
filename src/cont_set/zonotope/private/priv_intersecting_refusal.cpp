// priv_intersecting_refusal - why is_intersecting of a zonotope or a constrained zonotope refuses
//
// Syntax:   Refusal r = priv_intersecting_refusal(type, c, other);
// Inputs:   type - "exact" or "approx";  c, other - the centers of the two sets (a batch has more
//           than two dimensions)
// Outputs:  r - the identifier and text of the refusal: another type, or a batch (a linear program
//           per pair); none if the call is valid
// See also: Zonotope::is_intersecting, ConZonotope::is_intersecting
//
// CORA: contSet/@zonotope/isIntersecting_.m

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Refusal priv_intersecting_refusal(const std::string &type, const Tensor &c, const Tensor &other) {
    if (type != "exact" && type != "approx")
        return std::pair{Error::wrong_value,
                         "the type must be 'exact' or 'approx', got '" + type + "'"};
    if (c.ndim() != 2 || other.ndim() != 2)
        return std::pair{Error::not_supported, std::string("it supports single sets, not a batch")};
    return std::nullopt;
}

Refusal priv_intersecting_points_refusal(const std::string &type, const Tensor &c,
                                         const Tensor &p) {
    // points are columns (n, N) of the dimension n of the set
    if (p.ndim() != 2 || p.shape()[0] != c.shape()[c.ndim() - 2])
        return std::pair{Error::dimension_mismatch,
                         "the points must be a matrix (" + std::to_string(c.shape()[c.ndim() - 2]) +
                             ", N) of columns, got the shape " + shape_text(p.shape())};
    return priv_intersecting_refusal(type, c, p);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
