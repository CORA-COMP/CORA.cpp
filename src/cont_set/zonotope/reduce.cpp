// reduce - fewer generators for a zonotope, as CORA's zonotope.reduce with method "girard"
//
// The generators that cost the most to box stay, by Girard's metric ||g||_1 - ||g||_inf (zero on an
// axis-aligned one); the others are replaced by the box that encloses them. The result contains the
// zonotope and has at most order * n generators (n the dimension). A batch reduces member by
// member.
//
// Syntax:   Zred = Z.reduce(order);
// Inputs:   order - the largest number of generators per dimension, at least 1
// Outputs:  Zred - a zonotope that encloses Z; Z itself if it has few enough generators
// See also: interval, plus
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceGirard.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the zonotope order must be at least 1, got " + std::to_string(order);
    });
    const int64_t n = dim(), m = nr_generators();
    if (m <= order * n) return *this;

    // the longest generators stay and the rest becomes a box; a matrix zonotope as its entries
    if (elem_rank_ == 1) return {c_, priv_girard_generators(G_, order * n - n)};
    const Shape batch = batch_shape();
    Shape flat = batch, shape = G_.shape();
    flat.push_back(m);
    flat.push_back(n);
    const Tensor reduced = priv_girard_generators(G_.reshape(flat), order * n - n);
    shape[batch.size()] = reduced.size(-2);
    return {c_, reduced.reshape(shape), elem_rank_};
}

Set Zonotope::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
