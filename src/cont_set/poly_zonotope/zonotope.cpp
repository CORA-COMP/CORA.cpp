// zonotope - a zonotope that contains a polynomial zonotope, as CORA's polyZonotope.zonotope
//
// A generator whose exponents are all even only reaches [0, 1] of its length: half of it moves
// to the center, the other half stays a generator. The independent generators stay as they are.
//
// Syntax:   Z = pZ.zonotope();
// Inputs:   -
// Outputs:  Z - the enclosing zonotope, with the center c + half the sum of the even generators
// See also: interval, reduce
//
// CORA: contSet/@polyZonotope/zonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope PolyZonotope::zonotope() const {
    detail::Entry entry;
    const int64_t n = dim(), h = num_dependent();
    // an odd generator stays, an even one is halved, a constant one joins the center whole
    std::vector<int64_t> keep, even;
    std::vector<char> constant(std::size_t(h), 0);
    for (int64_t j = 0; j < h; ++j) {
        const std::vector<int> &col = E_[std::size_t(j)];
        if (std::all_of(col.begin(), col.end(), [](int e) { return e == 0; }))
            constant[std::size_t(j)] = 1;
        else
            (priv_is_even(col) ? even : keep).push_back(j);
    }
    if (even.empty() && keep.size() == std::size_t(h))
        return Zonotope(c_, Tensor::concat({G_, GI_}, -2));
    // the center takes the shift 0.5 * sum(Gquad), added from the last generator to the first
    const std::vector<double> g = G_.data(), cv = c_.data();
    std::vector<double> shifted(static_cast<std::size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        double shift = 0.0;
        for (int64_t j = h; j-- > 0;) {
            const double x = g[std::size_t(j * n + i)];
            shift = (constant[std::size_t(j)]           ? x
                     : priv_is_even(E_[std::size_t(j)]) ? 0.5 * x
                                                        : 0.0) +
                    shift;
        }
        shifted[std::size_t(i)] = cv[std::size_t(i)] + shift;
    }
    const Tensor G_even = even.empty() ? G_.zeros_like({0, n}) : priv_rows(G_, even) * 0.5;
    return Zonotope(Tensor::like(c_, shifted, {n}),
                    Tensor::concat({priv_rows(G_, keep), G_even, GI_}, -2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
