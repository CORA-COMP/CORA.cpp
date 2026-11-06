// priv_square_terms - the sums that the square and the second-order terms share
//
// Syntax:   auto [lo, hi] = priv_dependent_sums(A);   auto [s, S] = priv_diagonal_pair_sums(A);
// Inputs:   A - a square interval (..., n, n), the bounds of an interval matrix
// Outputs:  priv_dependent_sums: entry (i, j) is the sum over k other than i and j of the hull of
//           the products A(i, k) A(k, j), bounds (lo, hi);
//           priv_diagonal_pair_sums: entry (i, j) is A(i, i) + A(j, j), bounds (s, S)
// See also: exact_square, dependent_terms
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

std::pair<Tensor, Tensor> priv_dependent_sums(const Interval &A);

std::pair<Tensor, Tensor> priv_diagonal_pair_sums(const Interval &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
