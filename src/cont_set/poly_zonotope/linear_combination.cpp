// linear_combination - the polynomial zonotope of all (1 - t)/2 x + (1 + t)/2 y, as CORA's linComb
//
// The segments between the points of the two sets, as CORALean's linComb lays them out: the
// factors of the sets side by side (independent of each other, identifiers 1, 2, ... as CORA sets
// them) and one new factor t last. Dependent generators: half of those of each set (the first set
// at 1 and at t, the second set at 1 and at -t), the half-difference of the centers at t; the
// center is the midpoint. The independent generators pair as CORA's zonotope enclose does: the
// half-sums, then the half-differences, the surplus of the longer block copied. Nothing is merged
// and no zero generator dropped (CORA merges and moves constants to the center, same set).
//
// Syntax:   S = pZ.linear_combination(S2);
// Inputs:   S2 - polynomial zonotope, zonotope or box of the dimension of pZ
// Outputs:  S - the combination, with p1 + p2 + 1 factors
// See also: enclose, exact_plus
//
// CORA: contSet/@polyZonotope/linComb.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::linear_combination(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "linear_combination: the sets differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S.dim());
    });
    const std::size_t p1 = id_.size(), p2 = S.id_.size(), p = p1 + p2 + 1;
    const int64_t h1 = num_dependent(), h2 = S.num_dependent();
    // half of every generator, the second set's negated in the block that carries t
    const Tensor A = G_ * 0.5, B = S.G_ * 0.5;
    const Tensor halved = (c_ - S.c_) * 0.5;
    const Tensor G = Tensor::concat({A, B, A, -B, halved.expand_dims(-2)}, -2);
    // the factors of the sets side by side, t last: t is 0 in the first block, 1 in the second
    Exponents E;
    const auto column = [&](const std::vector<int> &e1, const std::vector<int> &e2, int t) {
        std::vector<int> col(p, 0);
        std::copy(e1.begin(), e1.end(), col.begin());
        std::copy(e2.begin(), e2.end(), col.begin() + std::ptrdiff_t(p1));
        col[p - 1] = t;
        return col;
    };
    const std::vector<int> none1(p1, 0), none2(p2, 0);
    for (int t = 0; t < 2; ++t) {
        for (int64_t j = 0; j < h1; ++j) E.push_back(column(E_[std::size_t(j)], none2, t));
        for (int64_t j = 0; j < h2; ++j) E.push_back(column(none1, S.E_[std::size_t(j)], t));
    }
    // the last column is the half-difference of the centers, at t
    E.push_back(column(none1, none2, 1));
    // the identifiers 1, 2, ..., as CORA sets them
    std::vector<int64_t> id(p);
    std::iota(id.begin(), id.end(), int64_t(1));
    return {(c_ + S.c_) * 0.5, G, priv_pair_independent(GI_, S.GI_), std::move(E), std::move(id),
            detail::Verbatim{}};
}

PolyZonotope PolyZonotope::linear_combination(const Zonotope &Z) const {
    detail::Entry entry;
    return linear_combination(PolyZonotope(Z));
}

PolyZonotope PolyZonotope::linear_combination(const Interval &I) const {
    detail::Entry entry;
    return linear_combination(PolyZonotope(I));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
