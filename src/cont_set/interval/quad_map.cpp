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
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

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
    CORA_CHECK(aux_has_zonotope(*this), Error::wrong_value,
               [] { return std::string("quad_map: the box must be a bounded vector"); });
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    return zonotope().quad_map(Q).interval();
}

Interval Interval::quad_map(const Interval &I2, const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    CORA_CHECK(aux_has_zonotope(*this) && aux_has_zonotope(I2), Error::wrong_value,
               [] { return std::string("quad_map: the boxes must be bounded vectors"); });
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "quad_map: the boxes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return zonotope().quad_map(I2.zonotope(), Q).interval();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
