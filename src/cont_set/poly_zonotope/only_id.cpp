// only_id - the part of a polynomial zonotope that depends on given factors, as CORA's onlyId
//
// The dependent generators whose exponents vanish outside ids (and the center) form the first set,
// over those factors; everything else, with the independent generators, the second, over all
// factors. Their sum has the points of the set.
//
// Syntax:   [pZ_id, pZ_r] = pZ.only_id(ids);
// Inputs:   ids - distinct identifiers of the set
// Outputs:  pZ_id - the center and the generators of the factors ids alone
//           pZ_r - the remaining generators (center 0), the independent ones included
// See also: rm_id, no_indep
//
// CORA: contSet/@polyZonotope/onlyId.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>
#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<PolyZonotope, PolyZonotope> PolyZonotope::only_id(const std::vector<int64_t> &ids) const {
    detail::Entry entry;
    CORA_CHECK(std::set<int64_t>(ids.begin(), ids.end()).size() == ids.size(), Error::wrong_value,
               [] { return std::string("only_id: ids must not contain duplicates"); });
    CORA_CHECK(
        std::all_of(ids.begin(), ids.end(), [&](int64_t v) { return priv_find_id(id_, v) >= 0; }),
        Error::wrong_value,
        [] { return std::string("only_id: ids must only contain identifiers of the set"); });
    // the factors in ids, in the order of the set
    std::vector<bool> inside(id_.size(), false);
    std::vector<int64_t> id_in;
    for (std::size_t i = 0; i < id_.size(); ++i)
        if (std::find(ids.begin(), ids.end(), id_[i]) != ids.end()) {
            inside[i] = true;
            id_in.push_back(id_[i]);
        }
    // a generator is in the first part if no factor outside ids has an exponent
    std::vector<int64_t> part, rest;
    Exponents E_in;
    for (std::size_t j = 0; j < E_.size(); ++j) {
        bool only = true;
        std::vector<int> col;
        for (std::size_t i = 0; i < id_.size(); ++i) {
            if (inside[i])
                col.push_back(E_[j][i]);
            else
                only = only && E_[j][i] == 0;
        }
        if (only) {
            part.push_back(int64_t(j));
            E_in.push_back(std::move(col));
        } else {
            rest.push_back(int64_t(j));
        }
    }
    // the second part keeps the other generators with their exponents over all factors
    Exponents E_rest;
    for (const int64_t j : rest) E_rest.push_back(E_[std::size_t(j)]);
    return {PolyZonotope(c_, priv_rows(G_, part), GI_.zeros_like({0, dim()}), std::move(E_in),
                         std::move(id_in)),
            PolyZonotope(c_.zeros_like(), priv_rows(G_, rest), GI_, std::move(E_rest), id_)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
