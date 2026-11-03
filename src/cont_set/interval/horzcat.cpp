// horzcat - intervals joined side by side, as CORA's interval.horzcat
//
// Syntax:   Interval I = Interval::horzcat({I1, I2, ...});
// Inputs:   Is - the intervals, which agree in the number of rows
// Outputs:  I - the bounds joined as columns; nothing rounds (a number is a point interval, made
//           by the caller)
// See also: cat, cart_prod
//
// CORA: contSet/@interval/horzcat.m

#include "cont_set/interval/private/priv_join.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::horzcat(const std::vector<Interval> &Is) {
    detail::Entry entry;
    CORA_CHECK(!Is.empty(), Error::wrong_value,
               [] { return std::string("horzcat: give at least one interval to join"); });
    for (const Interval &I : Is)
        CORA_CHECK(priv_joinable(Is[0].inf(), I.inf(), 2), Error::dimension_mismatch, [&] {
            return "horzcat: the bounds must agree in the rows, got the shapes " +
                   shape_text(Is[0].inf().shape()) + " and " + shape_text(I.inf().shape());
        });
    return priv_join(Is, 2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
