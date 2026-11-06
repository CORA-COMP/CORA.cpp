// tril - the lower triangular part of an interval matrix, as CORA's interval.tril
//
// Syntax:   Interval R = I.tril();   Interval R = I.tril(K);
// Inputs:   K - the diagonal from which on the entries are kept, K >= 0 (default 0)
// Outputs:  R - the entries on and below the K-th diagonal, [0, 0] above it
// See also: triu, Tensor::tril
//
// CORA: contSet/@interval/tril.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::tril(int64_t K) const {
    detail::Entry entry;
    CORA_CHECK(K >= 0, Error::wrong_value,
               [&] { return "tril: K must be nonnegative, got " + std::to_string(K); });
    return {inf_.tril(K), sup_.tril(K)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
