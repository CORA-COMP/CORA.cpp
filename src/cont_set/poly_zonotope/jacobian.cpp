// jacobian - the derivatives of a polynomial zonotope along its factors, as CORA's jacobian
//
// For every chosen factor k a polynomial zonotope of the derivative of the dependent part: the
// generator of every column is E[k, i] times the original, added up one copy after the other from
// the right (CORALean's rounding), the exponent of k lowered by one (a column of exponent 0 stays
// as a zero generator); the center is 0 and the independent generators are dropped. CORA moves
// the constants to the center and drops the zero columns (same point).
//
// Syntax:   D = pZ.jacobian();   D = pZ.jacobian(ids);
// Inputs:   ids - the identifiers of the factors (default: all, in the order of the set)
// Outputs:  D - one polynomial zonotope per factor, over the identifiers of the set
// See also: get_subset, split_dep_factor
//
// CORA: contSet/@polyZonotope/jacobian.m
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

std::vector<PolyZonotope>
PolyZonotope::jacobian(const std::optional<std::vector<int64_t>> &ids) const {
    detail::Entry entry;
    std::vector<int64_t> chosen = id_;
    if (ids) chosen = *ids;
    std::vector<PolyZonotope> out;
    out.reserve(chosen.size());
    for (const int64_t id : chosen) {
        const int64_t k = priv_find_id(id_, id);
        CORA_CHECK(k >= 0, Error::wrong_value, [&] {
            return "jacobian: the identifier " + std::to_string(id) + " is not in the set";
        });
        // the multiples: E[k, i] copies of generator i, added from the last copy to the first
        std::vector<Tensor> rows;
        rows.reserve(E_.size());
        Exponents E = E_;
        for (std::size_t i = 0; i < E_.size(); ++i) {
            Tensor sum = c_.zeros_like();
            const Tensor g = G_.slice(-2, int64_t(i), int64_t(i) + 1).squeeze(-2);
            for (int m = 0; m < E_[i][std::size_t(k)]; ++m) sum = g + sum;
            rows.push_back(sum.expand_dims(-2));
            E[i][std::size_t(k)] = std::max(E_[i][std::size_t(k)] - 1, 0);
        }
        const Tensor G = rows.empty() ? G_ : Tensor::concat(rows, -2);
        out.emplace_back(c_.zeros_like(), G, GI_.zeros_like({0, dim()}), std::move(E), id_,
                         detail::Verbatim{});
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
