// restructure - a polynomial zonotope without independent generators, as CORALean's restructure
//
// The independent generators are boxed into one generator per coordinate (the row sums of their
// magnitudes, added from the last generator to the first, as CORALean's reduceBy), every
// dependent generator is kept, and the boxes are made dependent: each gets a factor of its own
// (a fresh identifier above all of the set), as does a zero column per coordinate (CORALean's
// absorbRounding of the empty error box). The result has no independent generators and contains
// the set; the dependency of the kept generators is kept. CORA's restructure(pZ, method, order)
// instead builds a zonotope enclosure, reduced by a technique of the user's choice
// (docs/cora-differences.md).
//
// Syntax:   S = pZ.restructure();
// Inputs:   -
// Outputs:  S - a polynomial zonotope that contains the set
// See also: reduce, no_indep
//
// CORA: contSet/@polyZonotope/restructure.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::restructure() const {
    detail::Entry entry;
    const int64_t n = dim(), q = GI_.shape()[0];
    // the radius of the independent part, the sum from the last generator to the first
    Tensor radius = c_.zeros_like();
    for (int64_t j = q; j-- > 0;) radius = GI_.slice(-2, j, j + 1).squeeze(-2).abs() + radius;
    // the boxes and the zero columns of the error box, each on a factor of its own
    const int64_t k = 2 * n, p = int64_t(id_.size());
    // fresh identifiers above those of the set
    const int64_t fresh =
        (id_.empty() ? int64_t(0) : *std::max_element(id_.begin(), id_.end())) + 1;
    // the exponents of the old columns padded, then one unit column per new factor
    std::vector<int64_t> id = id_;
    for (int64_t j = 0; j < k; ++j) id.push_back(fresh + j);
    Exponents E;
    for (const std::vector<int> &col : E_) {
        std::vector<int> e = col;
        e.resize(std::size_t(p + k), 0);
        E.push_back(std::move(e));
    }
    for (int64_t j = 0; j < k; ++j) {
        std::vector<int> e(std::size_t(p + k), 0);
        e[std::size_t(p + j)] = 1;
        E.push_back(std::move(e));
    }
    // the boxes as diagonal generators, then the zero columns of the empty error box
    const Tensor zeros = c_.zeros_like({n, n});
    return {c_,
            Tensor::concat({G_, radius.expand_dims(-1).diag(), zeros}, -2),
            GI_.zeros_like({0, n}),
            std::move(E),
            std::move(id),
            detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
