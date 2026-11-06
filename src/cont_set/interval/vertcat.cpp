// vertcat - intervals stacked, as CORA's interval.vertcat
//
// A number is a point interval, made by the caller.
//
// Syntax:   Interval I = Interval::vertcat({I1, I2, ...});
// Inputs:   Is - the intervals, which agree in the number of columns
// Outputs:  I - the bounds stacked as rows; nothing rounds
// See also: horzcat, cat
//
// CORA: contSet/@interval/vertcat.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_join.h"
#include "cora/cont_set/interval/interval.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::vertcat(const std::vector<Interval> &Is) {
    detail::Entry entry;
    CORA_CHECK(!Is.empty(), Error::wrong_value,
               [] { return std::string("vertcat: give at least one interval to stack"); });
    const auto same_columns = [&](const Interval &I) {
        return priv_joinable(Is[0].inf(), I.inf(), 1);
    };
    CORA_CHECK(std::all_of(Is.begin(), Is.end(), same_columns), Error::dimension_mismatch,
               [] { return std::string("vertcat: the bounds must agree in the columns"); });
    return priv_join(Is, 1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
