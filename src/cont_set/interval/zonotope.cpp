// zonotope - the zonotope of a box, as CORA's interval.zonotope
//
// Syntax:   Zonotope Z = I.zonotope();
// Inputs:   -
// Outputs:  Z - the center c of the box with the generators diag(r), r = max(sup - c, c - inf)
//           (the radius, reaching both ends from the rounded center, as CORALean's zonotope); the
//           columns of zero radius are dropped, except for a batch (the ranks may differ), which
//           keeps all n, the same set. Error::not_supported for an empty interval
// See also: center, rad, Zonotope::interval
//
// CORA: contSet/@interval/zonotope.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Interval::zonotope() const {
    detail::Entry entry;
    CORA_CHECK(!is_empty(), Error::not_supported,
               [] { return std::string("zonotope: the empty interval has no zonotope here"); });
    CORA_CHECK(inf_.size(-1) == 1, Error::wrong_value, [&] {
        return "zonotope: the interval must not be an n-d array with n > 1, got the shape " +
               shape_text(inf_.shape());
    });
    const Tensor c = center();
    const Tensor r = (sup_ - c).maximum(c - inf_);
    const Tensor G = r.diag();
    if (inf_.shape().size() > 2) return {c, G};
    return {c, G.take(r.not_equal(r.zeros_like()).reshape({dim()}).nonzero(), -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
