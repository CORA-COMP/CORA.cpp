// quad_map - the box around a quadratic map of boxes, as CORA's interval.quadMap
//
// Syntax:   Ires = I.quad_map(Q);   Ires = I.quad_map(I2, Q);
// Inputs:   Q - the matrices of the outputs, Q[i] (n, n): output i is x' Q[i] x (or x1' Q[i] x2
//           for x1 in I and x2 in I2); the boxes are bounded vectors
// Outputs:  Ires - the interval hull of the quad_map of the zonotopes of the boxes, one
//           coordinate per matrix
// See also: Zonotope::quad_map
//
// CORA: contSet/@interval/quadMap.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the box is a bounded vector, the only one that has a zonotope.
bool aux_has_zonotope(const Interval &I) { return I.is_bounded() && I.inf().shape().back() == 1; }

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::quad_map(const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    CORA_CHECK(aux_has_zonotope(*this), Error::wrong_value, [] {
        return std::string("quad_map: the box must be a bounded vector");
    });
    return priv_zonotope_of_interval(*this).quad_map(Q).interval();
}

Interval Interval::quad_map(const Interval &I2, const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    CORA_CHECK(aux_has_zonotope(*this) && aux_has_zonotope(I2), Error::wrong_value, [] {
        return std::string("quad_map: the boxes must be bounded vectors");
    });
    return priv_zonotope_of_interval(*this)
        .quad_map(priv_zonotope_of_interval(I2), Q)
        .interval();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
