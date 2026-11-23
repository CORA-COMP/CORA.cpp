// resolve - a polynomial zonotope with factors set to values, as CORA's resolve
//
// The factors ids take the values x, one factor after the other in ascending identifier order, as
// CORALean's resolve (get_subset at [x, x] per factor): every dependent generator is scaled by the
// power of the value, the factors leave, the generators that no longer depend on a factor join the
// center, and equal exponents merge. CORA returns a vector for a set without generators, here it
// is a PolyZonotope with none.
//
// Syntax:   S = pZ.resolve(x);   S = pZ.resolve(x, ids);
// Inputs:   x - a value per identifier (n entries)
//           ids - identifiers of the set (default: all of them, in order)
// Outputs:  S - the set over the other factors (the identifiers that stay keep their order)
// See also: get_subset, subs
//
// CORA: contSet/@polyZonotope/resolve.m
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

PolyZonotope PolyZonotope::resolve(const Tensor &x,
                                   const std::optional<std::vector<int64_t>> &ids) const {
    detail::Entry entry;
    const std::vector<int64_t> id = ids ? *ids : id_;
    CORA_CHECK(x.ndim() == 1 && x.shape()[0] == int64_t(id.size()), Error::wrong_value, [] {
        return std::string("resolve: the length of ids should match the length of x");
    });
    CORA_CHECK(
        std::set<int64_t>(id.begin(), id.end()).size() == id.size() &&
            std::all_of(id.begin(), id.end(), [&](int64_t v) { return priv_find_id(id_, v) >= 0; }),
        Error::wrong_value, [] { return std::string("resolve: invalid ids"); });
    // the factors in ascending order of their identifier, each scaled in turn as get_subset does
    std::vector<std::size_t> order(id.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return id[a] < id[b]; });
    std::vector<int64_t> sorted_id;
    std::vector<Tensor> sorted_x;
    for (const std::size_t k : order) {
        sorted_id.push_back(id[k]);
        sorted_x.push_back(x.slice(-1, int64_t(k), int64_t(k) + 1));
    }
    if (sorted_id.empty()) return get_subset({}, x);
    return get_subset(sorted_id, Tensor::concat(sorted_x, -1));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
