// split_first_gen - the two halves along the generator that is least orthogonal to the others
//
// Syntax:   [Z1, Z2] = Z.split_first_gen();
// Inputs:   -
// Outputs:  Z1, Z2 - the halves along generator j with the largest score sum_k |g_j' g_k| (the
//           first of equal ones), centers c - g_j/2 and c + g_j/2: they cover Z. CORA passes j to
//           a split by coordinate of the box around Z, so its halves belong to another coordinate;
//           CORA.cpp splits generator j. Without generators both are Z. CORA's cell array of
//           zonotopes is a loop over this. Not a batch
// See also: split, split_longest_gen
//
// CORA: contSet/@zonotope/splitFirstGen.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Zonotope, Zonotope> Zonotope::split_first_gen() const {
    detail::Entry entry;
    CORA_CHECK(G_.ndim() == 2, Error::not_supported, [] {
        return std::string("split_first_gen: a generator per zonotope: it supports a single "
                           "zonotope, not a batch");
    });
    if (G_.shape().back() == 0) return {*this, *this};
    // the choice is made on detached values and carries no gradient
    const Tensor values = G_.detach();
    const Tensor score = values.matrix_transpose().matmul(values).abs().sum(-1);
    return priv_split_generator(*this, int64_t(score.argmax(0).item()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
