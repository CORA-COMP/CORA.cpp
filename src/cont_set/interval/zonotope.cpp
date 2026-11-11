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
    const std::vector<int64_t> axes = any_radius.nonzero();
    if (r.ndim() == 2 && !r.requires_grad()) {
        // one interval (n, 1): the generators of its axes directly, not through a dense n x n
        // diagonal (125 GB for the 125 000 states of the HEAT04 benchmark)
        const int64_t n = r.shape()[0], k = static_cast<int64_t>(axes.size());
        const std::vector<double> radius = r.data();
        std::vector<double> G(static_cast<size_t>(n * k), 0.0);
        for (int64_t j = 0; j < k; ++j)
            G[static_cast<size_t>(axes[static_cast<size_t>(j)] * k + j)] =
                radius[static_cast<size_t>(axes[static_cast<size_t>(j)])];
        return {c, k == 0 ? r.zeros_like({n, 0}) : Tensor::like(r, G, {n, k})};
    }
    return {c, priv_select_generators(r.diag(), axes)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
