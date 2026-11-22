// priv_intersecting_refusal - why is_intersecting of a zonotope or a constrained zonotope refuses
//
// A batch is refused because it needs a linear program per pair.
//
// Syntax:   Refusal r = priv_intersecting_refusal(method, c, other);
// Inputs:   method - "exact" or "approx"
//           c, other - the centers of the two sets (a batch has more than two dimensions)
// Outputs:  r - the identifier and text of the refusal (another method, a batch); none if valid
// See also: Zonotope::is_intersecting, ConZonotope::is_intersecting
//
// CORA: contSet/@zonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Refusal priv_intersecting_refusal(const std::string &method, const Tensor &c, const Tensor &other) {
    if (method != "exact" && method != "approx")
        return std::pair{Error::wrong_value,
                         "the method must be 'exact' or 'approx', got '" + method + "'"};
    if (c.ndim() != 2 || other.ndim() != 2)
        return std::pair{Error::not_supported, std::string("it supports single sets, not a batch")};
    return std::nullopt;
}

Refusal priv_intersecting_points_refusal(const std::string &method, const Tensor &c,
                                         const Tensor &p) {
    // points are rows (..., n) of the dimension n of the set, whose center c is (..., n)
    const int64_t n = c.size(-1);
    if (!is_vector(p, n))
        return std::pair{Error::dimension_mismatch,
                         "the points must be (..., " + std::to_string(n) +
                             "), one per row, got the shape " + shape_text(p.shape())};
    // any number of points: only the set must be single
    return priv_intersecting_refusal(method, c.expand_dims(-1), c.expand_dims(-1));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
