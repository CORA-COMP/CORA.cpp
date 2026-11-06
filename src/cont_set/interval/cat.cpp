// cat - intervals joined along a dimension, as CORA's cat for intervals
//
// Syntax:   Interval I = Interval::cat(dim, {I1, I2, ...});
// Inputs:   dim - 1 (stack the rows) or 2 (join the columns)
//           Is - the intervals, which agree in the other dimensions
// Outputs:  I - the interval of the concatenated bounds; nothing rounds
// See also: cart_prod
//
// CORA: contSet/@interval/cat.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_join.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::cat(int dim, const std::vector<Interval> &Is) {
    detail::Entry entry;
    CORA_CHECK(dim >= 1, Error::wrong_value,
               [&] { return "cat: dim must be positive, got " + std::to_string(dim); });
    CORA_CHECK(dim <= 2, Error::not_supported, [&] {
        return "cat: dim " + std::to_string(dim) +
               " needs n-d arrays; the leading dimensions of "
               "CORA.cpp's bounds are batches, so only dim 1 and 2 are supported";
    });
    CORA_CHECK(!Is.empty(), Error::wrong_value,
               [] { return std::string("cat: give at least one interval to join"); });
    for (const Interval &I : Is)
        CORA_CHECK(priv_joinable(Is[0].inf(), I.inf(), dim), Error::dimension_mismatch, [&] {
            return "cat: the bounds must agree in all dimensions but " + std::to_string(dim) +
                   ", got the shapes " + shape_text(Is[0].inf().shape()) + " and " +
                   shape_text(I.inf().shape());
        });
    return priv_join(Is, dim);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
