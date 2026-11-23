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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
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

/// CORA's hash vector rand(1, p) of RandStream('mt19937ar', 'Seed', 0), which
/// removeRedundantExponents multiplies with the exponents to order the columns.
std::vector<double> priv_hash_vector(std::size_t p);

/// CORA's removeRedundantExponents: equal exponents add up (last row to first), zero generators
/// go, the generator rows are in the order of CORA's hash (singles first, then the sums).
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

/// The rows of G (h, n) at the indices, in that order (k, n).
Tensor priv_rows(const Tensor &G, const std::vector<int64_t> &index);

/// For each exponent column, the least index of a column equal to it (a column that is the first of
/// its kind is its own).
std::vector<int64_t> priv_least_equal_column(const Exponents &E);

/// Whether all exponents of the column are even (such a generator only reaches [0, 1]).
bool priv_is_even(const std::vector<int> &exponents);

/// The set with its generators of exponent 0 (a constant) moved into the center.
PolyZonotope priv_fold_constants(const PolyZonotope &set);

/// CORA's ending of getSubset: the columns of equal exponents merged, the generators of exponent 0
/// added to the center, the identifiers no generator uses dropped.
PolyZonotope priv_merged_set(const Tensor &c, const Tensor &G, const Tensor &GI, const Exponents &E,
                             const std::vector<int64_t> &id);

/// The position of the identifier in ids, -1 if it is not there.
int64_t priv_find_id(const std::vector<int64_t> &ids, int64_t id);

/// The identifiers ids followed by `count` new ones, each above all of ids (the next integers).
std::vector<int64_t> priv_append_fresh_ids(const std::vector<int64_t> &ids, int64_t count);

/// The exponent columns of [c, G]: a column of zeros for the center, then those of E, over `rows`
/// identifiers.
Exponents priv_extended_exponents(const Exponents &E, std::size_t rows);

/// CORALean's absorbRounding at an empty error box: the independent generators and one zero
/// column per coordinate become dependent generators, each on a new factor. Nothing is merged.
PolyZonotope priv_absorb_rounding(const PolyZonotope &set);

/// The independent block of a linear combination (CORALean's pairSum then pairDiff): the half-sums
/// of the rows of A and B, then the half-differences, the surplus rows of the longer one copied.
Tensor priv_pair_independent(const Tensor &A, const Tensor &B);

/// The matrix product A B (..., r, k) x (..., k, c) with every entry summed from the last term to
/// the first (CORALean's dotDown, sumDown nests to the right), on every backend: the batch axes
/// broadcast, a term is one product, and the sum starts at zero.
Tensor priv_matmul_nested(const Tensor &A, const Tensor &B);

/// The coordinatewise product of two sets over the same identifiers (CORALean's mul at nearest):
/// a generator per pair of columns of [c, G], the first slower, exponents added, center 0. Neither
/// set has independent generators.
PolyZonotope priv_multiply(const PolyZonotope &a, const PolyZonotope &b);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
