// support_function - the support function of a polygon, as CORA's supportFunc_
//
// The largest of the pieces' bounds of d . x (CORALean's Polygon.boundDot: the fold of
// maximum over the pieces, seeded with the first piece's), each the largest d . v over the
// vertices; -Inf for no piece (CORALean has 0 there, as no set is bounded).
//
// Syntax:   Tensor s = pgon.support_function(d);
// Inputs:   d - the direction (2)
// Outputs:  s - max_{x in pgon} d'x, a number
// See also: interval, priv_bound_dot
//
// CORA: contSet/@polygon/supportFunc_.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Polygon::support_function_impl(const Tensor &d) const {
    const std::vector<double> a = d.data();
    double best = -std::numeric_limits<double>::infinity();
    if (!pieces_.empty()) {
        const std::vector<Piece> hosts = priv_hosts(*this);
        const Point dir = {a[0], a[1]};
        // foldr maximum over the pieces, seeded with the first one's bound
        best = priv_bound_dot(hosts.front(), dir);
        for (std::size_t i = hosts.size(); i-- > 0;)
            best = priv_maximum(priv_bound_dot(hosts[i], dir), best);
    }
    return Tensor::like(d, {best}, {});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
