// stack - a batch of zonotopes as one zonotope
//
// Syntax:   Z = Zonotope::stack({Z1, Z2, ...});
// Inputs:   Zs - zonotopes of one dimension and batch shape; the generator counts may differ
// Outputs:  Z - one zonotope whose leading batch dimension indexes Zs, in order
// See also: plus, contSet/interval/stack
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// G with zero generators appended up to m, along the generator axis.
Tensor aux_pad_generators(const Tensor &G, int64_t m, int axis) {
    std::vector<int64_t> shape = G.shape();
    const int64_t missing = m - shape[axis];
    if (missing == 0) return G;
    shape[axis] = missing;
    return Tensor::concat({G, G.zeros_like(shape)}, axis);
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
    for (const Zonotope &Z : Zs)
        CORA_CHECK(Z.elem_rank() == Zs[0].elem_rank(), Error::dimension_mismatch,
                   [] { return std::string("stack: the zonotopes differ in element rank"); });
    // every member padded to the most generators, on the generator axis
    const int axis = int(Zs[0].center().ndim() - Zs[0].elem_rank());
    int64_t m = 0;
    for (const Zonotope &Z : Zs) m = std::max(m, Z.nr_generators());
    std::vector<Tensor> cs, Gs;
    for (const Zonotope &Z : Zs) {
        cs.push_back(Z.center());
        Gs.push_back(aux_pad_generators(Z.generators(), m, axis));
    }
    // the new batch axis comes first
    if (Zs[0].elem_rank() == 1) return {Tensor::stack(cs), Tensor::stack(Gs)};
    return matrix(Tensor::stack(cs), Tensor::stack(Gs));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
