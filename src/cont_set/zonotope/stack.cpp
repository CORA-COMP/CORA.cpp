// stack - a batch of zonotopes as one zonotope
//
// Syntax:   Z = Zonotope::stack({Z1, Z2, ...});
// Inputs:   Zs - zonotopes of one dimension and batch shape; the generator counts may differ
// Outputs:  Z - one zonotope whose leading batch dimension indexes Zs, in order
// See also: plus, contSet/interval/stack
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// G with zero columns appended up to m generators.
Tensor aux_pad_generators(const Tensor &G, int64_t m) {
    std::vector<int64_t> shape = G.shape();
    const int64_t missing = m - shape.back();
    if (missing == 0) return G;
    shape.back() = missing;
    return Tensor::concat({G, G.zeros_like(shape)}, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::stack(const std::vector<Zonotope> &Zs) {
    detail::Entry entry;
    CORA_CHECK(!Zs.empty(), Error::wrong_value,
               [] { return std::string("stack: give at least one zonotope"); });
    for (const Zonotope &Z : Zs)
        CORA_CHECK(Z.center().shape() == Zs[0].center().shape(), Error::dimension_mismatch, [&] {
            return "stack: the zonotopes must have one dimension and batch shape, got the "
                   "centers " +
                   shape_text(Zs[0].center().shape()) + " and " + shape_text(Z.center().shape());
        });
    int64_t m = 0;
    for (const Zonotope &Z : Zs) m = std::max(m, Z.generators().shape().back());
    std::vector<Tensor> cs, Gs;
    for (const Zonotope &Z : Zs) {
        cs.push_back(Z.center());
        Gs.push_back(aux_pad_generators(Z.generators(), m));
    }
    return {Tensor::stack(cs), Tensor::stack(Gs)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
