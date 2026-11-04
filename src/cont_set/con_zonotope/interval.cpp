// interval - the interval hull of a constrained zonotope, as CORA's interval(cz)
//
// Syntax:   Interval I = cz.interval();
// Inputs:   -
// Outputs:  I - the smallest box containing the set: without constraints the box of the zonotope,
//           else the exact range of every coordinate by two linear programs (the empty box for an
//           empty set). Not a batch if there are constraints
// See also: center, Zonotope::interval
//
// CORA: contSet/@conZonotope/interval.m

#include "cont_set/con_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval ConZonotope::interval_impl() const {
    if (A_.shape()[A_.ndim() - 2] == 0) return Zonotope(c_, G_).interval();
    // CORA drops the zero generators first; the ranges do not change
    const HostConZonotope H = priv_host_con_zonotope(priv_compact(*this));
    std::vector<double> inf(std::size_t(H.n)), sup(std::size_t(H.n));
    for (int64_t i = 0; i < H.n; ++i) {
        std::vector<double> direction(std::size_t(H.n), 0.0);
        direction[i] = 1.0;
        const auto range = priv_range(H, direction);
        if (!range) return Interval::empty(H.n);
        inf[i] = range->first;
        sup[i] = range->second;
    }
    return {Tensor::like(c_, inf, {H.n, 1}), Tensor::like(c_, sup, {H.n, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
