// lift - the set with its coordinates permuted or selected, as CORA's lift_ and project
//
// lift places state i at position dims[i] of N = n dimensions (CORA's lift_ of a permutation,
// CORALean's reindex); project keeps the coordinates dims (CORALean's project). The factors and
// the constraint stay.
//
// Syntax:   S = cPZ.lift(N, dims);   S = cPZ.project(dims);
// Inputs:   N - the dimension (n)
//           dims - the positions, 0-based
// Outputs:  S - the set in the new coordinates
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/lift_.m
// CORA: contSet/@conPolyZono/project.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"
#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set of the coordinates idx of every column (c, G, GI).
ConPolyZono aux_take(const ConPolyZono &S, const std::vector<int64_t> &idx) {
    return {PolyZonotope(S.c().take(idx, 0), S.generators().take(idx, 1),
                         S.independent_generators().take(idx, 1), S.exponents(), S.ids(),
                         detail::Verbatim{}),
            S.constraints()};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    std::vector<int64_t> sorted = dims;
    std::sort(sorted.begin(), sorted.end());
    bool perm = N == dim() && int64_t(dims.size()) == N;
    for (int64_t i = 0; perm && i < N; ++i) perm = sorted[std::size_t(i)] == i;
    CORA_CHECK(perm, Error::not_supported, [] {
        return std::string("lift: only a permutation of the coordinates (N = n), as CORALean's "
                           "reindex");
    });
    // the inverse permutation: position dims[i] reads state i
    std::vector<int64_t> inv(dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) inv[std::size_t(dims[i])] = int64_t(i);
    return aux_take(*this, inv);
}

Set ConPolyZono::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

Set ConPolyZono::project_impl(const std::vector<int64_t> &dims) const {
    return aux_take(*this, dims);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
