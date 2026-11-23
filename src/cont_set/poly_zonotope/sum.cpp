// sum - the sum of polynomial zonotopes over shared factors, as CORA's sum
//
// Equal identifiers are one factor in all summands: the generators and exponents join over the
// sorted union of the identifiers (columns in the order this set, then the list; equal exponents
// are not merged), the centers add as CORALean's sum: the center of this set plus the centers of
// the list added from the last entry to the first. CORA adds the center of the first list entry
// once per entry and places the exponents of the later sets by the union of only two id lists
// (docs/cora-differences.md); this is the correct sum.
//
// Syntax:   S = pZ.sum(list);
// Inputs:   list - polynomial zonotopes of the dimension of pZ
// Outputs:  S - the sum, as exact_plus of all of them
// See also: exact_plus, plus
//
// CORA: contSet/@polyZonotope/sum.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::sum(const std::vector<PolyZonotope> &list) const {
    detail::Entry entry;
    for (const PolyZonotope &S : list)
        CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
            return "sum: the sets differ in dimension, " + std::to_string(dim()) + " and " +
                   std::to_string(S.dim());
        });
    // the identifiers of all summands, sorted
    std::vector<int64_t> id = id_;
    for (const PolyZonotope &S : list) id = priv_merge_ids(id, S.id_).id;
    const auto extended = [&](const PolyZonotope &S) {
        return priv_extend_exponents(S.E_, priv_merge_ids(id, S.id_).position2, id.size());
    };
    Exponents E = priv_extend_exponents(E_, priv_merge_ids(id, id_).position2, id.size());
    std::vector<Tensor> G = {G_}, GI = {GI_};
    Tensor c_sum;
    for (const PolyZonotope &S : list) {
        const Exponents Ej = extended(S);
        E.insert(E.end(), Ej.begin(), Ej.end());
        G.push_back(S.G_);
        GI.push_back(S.GI_);
    }
    // the centers of the list nested to the right, as CORALean's sumDown
    for (std::size_t j = list.size(); j-- > 0;)
        c_sum = c_sum.defined() ? list[j].c_ + c_sum : list[j].c_;
    return {c_sum.defined() ? c_ + c_sum : c_, Tensor::concat(G, -2), Tensor::concat(GI, -2),
            std::move(E), std::move(id)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
