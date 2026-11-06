// support_function - the support function of a constrained zonotope
//
// Without constraints the formula of the zonotope; with constraints a linear program over the
// factors. The empty set has the value -Inf. A set with constraints is not a batch.
//
// Syntax:   rho = cz.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in cz} d'x, shape (..., 1, 1)
// See also: priv_support, Zonotope::support_function
//
// CORA: contSet/@conZonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ConZonotope::support_function_impl(const Tensor &d) const {
    if (A_.shape()[A_.ndim() - 2] == 0) return Zonotope(c_, G_).support_function(d);
    if (d.ndim() != 2)
        CORA_FAIL(Error::not_supported, "support_function: a constrained set takes one direction");
    const std::optional<PrivSupport> best = priv_support(priv_host_con_zonotope(*this), d.data());
    const double value = best ? best->value : -std::numeric_limits<double>::infinity();
    return Tensor::like(c_, {value}, {1, 1});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
