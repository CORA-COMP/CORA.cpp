// zonotope - the zonotope of a box, as CORA's interval.zonotope
//
// The radius reaches both ends from the rounded center, as CORALean's zonotope. The columns of
// zero radius are dropped (those of zero radius in every member of a batch). Error::not_supported
// for an empty interval.
//
// Syntax:   Zonotope Z = I.zonotope();
// Inputs:   -
// Outputs:  Z - the center c of the box with the generators diag(r), r = max(sup - c, c - inf)
// See also: center, Zonotope::interval
//
// CORA: contSet/@interval/zonotope.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/interval/interval.h"
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
    // the axes with a radius (in some member of a batch), by their index
    Tensor any_radius = r.detach().not_equal(r.zeros_like(r.shape()));
    while (any_radius.ndim() > 2) any_radius = any_radius.any(0);
    return {c, priv_select_generators(r.diag(), any_radius.nonzero())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
