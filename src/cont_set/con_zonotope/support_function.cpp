// support_function - the support function of a constrained zonotope
//
// Without constraints the formula of the zonotope; with constraints a linear program over the
// factors. The empty set has the value -Inf. A set with constraints is not a batch.
//
// Syntax:   rho = cz.support_function(d);   I = cz.support_function_range(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in cz} d'x, shape (..., 1, 1), or the range [min, max] as a box
// See also: priv_support, Zonotope::support_function
//
// CORA: contSet/@conZonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

#include <limits>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ConZonotope::support_function_impl(const Tensor &d) const {
    if (A().shape()[A().ndim() - 2] == 0) return Zonotope(c_, G_).support_function(d);
    if (d.ndim() != 2)
        CORA_FAIL(Error::not_supported, "support_function: a constrained set takes one direction");
    const std::optional<PrivSupport> best = priv_support(priv_host_con_zonotope(*this), d.data());
    const double value = best ? best->value : -std::numeric_limits<double>::infinity();
    return Tensor::like(c_.expand_dims(-1), {value}, {1, 1});
}

Interval ConZonotope::support_function_range(const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(d.ndim() == 2 && d.shape()[0] == dim() && d.shape()[1] == 1,
               Error::dimension_mismatch, [&] {
                   return "support_function_range: the direction must be a column (" +
                          std::to_string(dim()) + ", 1), got " + shape_text(d.shape());
               });
    // both ends from the programs over the factors; none if the set is empty
    const auto range = priv_range(priv_host_con_zonotope(*this), d.data());
    if (!range) return Interval::empty(1);
    return {Tensor::like(c_.expand_dims(-1), {range->first}, {1, 1}),
            Tensor::like(c_.expand_dims(-1), {range->second}, {1, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
