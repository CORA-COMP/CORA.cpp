// priv - the private functions of poly_zonotope, as CORA's @polyZonotope/private and its helpers
//
// Not for callers of the class: the operations of the polynomial zonotope use them. They work on
// the exponent matrices and the identifiers on the host; the generators stay tensors.
//
// Syntax:   MergedColumns m = priv_merge_columns(G, E);   Union u = priv_merge_ids(id1, id2);
// See also: poly_zonotope.h
//
// CORA: global/functions/helper/sets/contSet/polyZonotope/removeRedundantExponents.m
// CORA: global/functions/helper/sets/contSet/polyZonotope/mergeExpMatrix.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The generators and exponents after the merge of equal columns.
struct MergedColumns {
    Tensor G;
    Exponents E;
};

/// CORA's removeRedundantExponents: the generators with equal exponents add up (in the place of
/// the first of them) and zero generators go. CORA orders by a hash of the exponents, this keeps
/// the order of first occurrence.
MergedColumns priv_merge_columns(const Tensor &G, const Exponents &E);

/// The identifiers of two sets as one sorted list, and for each set the position of its
/// identifiers in it (CORA's mergeExpMatrix).
struct MergedIds {
    std::vector<int64_t> id;
    std::vector<std::size_t> position1, position2;
};
MergedIds priv_merge_ids(const std::vector<int64_t> &id1, const std::vector<int64_t> &id2);

/// The exponents E over the merged identifiers: the exponent of an identifier that the set does
/// not have is 0. `position` is the place of each of its identifiers.
Exponents priv_extend_exponents(const Exponents &E, const std::vector<std::size_t> &position,
                                std::size_t rows);

/// Removes the identifiers that no generator uses (all exponents 0): E and id shrink.
void priv_drop_unused_ids(Exponents &E, std::vector<int64_t> &id);

/// The columns of G (n, h) at the indices, in that order (n, k).
Tensor priv_columns(const Tensor &G, const std::vector<int64_t> &index);

/// Whether all exponents of the column are even (such a generator only reaches [0, 1]).
bool priv_is_even(const std::vector<int> &exponents);

/// The set with its generators of exponent 0 (a constant) moved into the center.
PolyZonotope priv_fold_constants(const PolyZonotope &set);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
