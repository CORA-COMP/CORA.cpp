// reduce - fewer generators for a constrained zonotope, as CORA's conZonotope.reduce
//
// Syntax:   cZred = cz.reduce(order);   cZred = cz.reduce(order, "girard");
// Inputs:   order - the generator budget per dimension, at least 1;  method - "girard", the one
//           of CORA's zonotope reductions that is ported; the others are Error::not_supported
// Outputs:  cZred - a constrained zonotope that encloses cz: the set without zero rows and
//           generators, and if it has more than order n + p generators (n the dimension, p the
//           constraints), Girard's rule applied to the zonotope [G; A] the constraints lift it to:
//           (order - 1) n generators stay and the others become one box generator per row of [G;
//           A], so the constraint rows get new free factors. CORA states the budget as the order
//           (order n + p) / (n + p) of the lifted zonotope in floating point; here it is the
//           integers, as CORALean's reduce. A batch reduces member by member
// See also: compact, Zonotope::reduce
//
// CORA: contSet/@conZonotope/reduce.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::reduce(int order, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the order must be at least 1, got " + std::to_string(order);
    });
    CORA_CHECK(method == "girard", Error::not_supported,
               [&] { return "reduce: the method '" + method + "' is not ported; use 'girard'"; });
    const ConZonotope cz = priv_compact(*this);
    const int64_t n = cz.dim(), p = cz.A().shape()[cz.A().ndim() - 2];
    if (cz.generators().shape().back() <= order * n + p) return cz;

    // the zonotope [G; A] with the center [c; -b] is reduced; its rows are read back
    const Tensor lifted = Tensor::concat({cz.generators(), cz.A()}, -2);
    const Tensor reduced = priv_girard_generators(lifted, (order - 1) * n);
    return {cz.c(), reduced.slice(-2, 0, n), reduced.slice(-2, n, n + p), cz.b()};
}

Set ConZonotope::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
