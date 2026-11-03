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
    const int64_t n = dim(), m = G_.shape().back();
    if (m <= order * n) return *this;

    // The kept generators come first, by Girard's score (the choice carries no gradient), then
    // one box generator per dimension for the rest; per member of a batch.
    const int64_t keep = order * n - n;
    const auto [kept, rest] = G_.detach().girard_split(keep);
    const Tensor box = G_.take_along_axis(rest, -1).abs_sum().diag();
    return {c_, keep == 0 ? box : Tensor::concat({G_.take_along_axis(kept, -1), box}, -1)};
}

Set Zonotope::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
