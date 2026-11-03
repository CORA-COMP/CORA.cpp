// triu - the upper triangular part of an interval matrix, as CORA's interval.triu
//
// Syntax:   Interval R = I.triu();   Interval R = I.triu(K);
// Inputs:   K - the diagonal from which on the entries are kept, K >= 0 (default 0)
// Outputs:  R - the entries on and above the K-th diagonal, [0, 0] below it
// See also: tril, Tensor::triu
//
// CORA: contSet/@interval/triu.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::triu(int64_t K) const {
    detail::Entry entry;
    CORA_CHECK(K >= 0, Error::wrong_value,
               [&] { return "triu: K must be nonnegative, got " + std::to_string(K); });
    return {inf_.triu(K), sup_.triu(K)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
