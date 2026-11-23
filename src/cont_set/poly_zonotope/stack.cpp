// stack - the product of polynomial zonotopes that share factors, as CORA's stack
//
// The sets are placed one below the other in pairs: two sets of dimensions n1 and n2 are padded
// with zeros to n1 + n2 coordinates and added by exact_plus, so that equal identifiers stay one
// factor; a set without dependent generators is simply added to the center. The pairs (first
// and second, third and fourth, ...) are then stacked in turn.
//
// Syntax:   S = PolyZonotope::stack({pZ1, pZ2, ...});
// Inputs:   sets - one or more polynomial zonotopes
// Outputs:  S - the stacked set of the sum of the dimensions
// See also: exact_plus, cart_prod
//
// CORA: contSet/@polyZonotope/stack.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of the matrix (rows, n) with `before` zero columns in front and `after` behind.
Tensor aux_pad(const Tensor &M, int64_t before, int64_t after) {
    const int64_t rows = M.shape()[0];
    return Tensor::concat({M.zeros_like({rows, before}), M, M.zeros_like({rows, after})}, -1);
}

/// The set with `before` zero coordinates in front and `after` behind.
PolyZonotope aux_extend(const PolyZonotope &S, int64_t before, int64_t after) {
    const Tensor &c = S.center();
    return {Tensor::concat({c.zeros_like({before}), c, c.zeros_like({after})}, -1),
            aux_pad(S.generators(), before, after),
            aux_pad(S.independent_generators(), before, after), S.exponents(), S.ids()};
}

/// CORA's aux_stack2: the two sets one below the other.
PolyZonotope aux_stack2(const PolyZonotope &A, const PolyZonotope &B) {
    const PolyZonotope a = aux_extend(A, 0, B.dim()), b = aux_extend(B, A.dim(), 0);
    if (a.num_dependent() == 0)
        return {b.center() + a.center(), b.generators(), b.independent_generators(), b.exponents(),
                b.ids()};
    if (b.num_dependent() == 0)
        return {a.center() + b.center(), a.generators(), a.independent_generators(), a.exponents(),
                a.ids()};
    return a.exact_plus(b);
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::stack(const std::vector<PolyZonotope> &sets) {
    detail::Entry entry;
    CORA_CHECK(!sets.empty(), Error::wrong_value,
               [] { return std::string("stack: at least one set is needed"); });
    std::optional<PolyZonotope> result;
    for (std::size_t i = 0; i < sets.size(); i += 2) {
        const PolyZonotope pair = i + 1 < sets.size() ? aux_stack2(sets[i], sets[i + 1]) : sets[i];
        result = result ? aux_stack2(*result, pair) : pair;
    }
    return *result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
