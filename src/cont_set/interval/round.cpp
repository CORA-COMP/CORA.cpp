// round - the bounds of a box rounded to digits decimals, as CORA's interval.round
//
// Syntax:   Interval R = I.round();   Interval R = I.round(digits);
// Inputs:   digits - the number of decimals, 0 (default) or more;  type - "decimals" (default);
//           "significant" is refused (Error::not_supported)
// Outputs:  R - [round(inf), round(sup)]: each bound to the nearest multiple of 10^-digits, halves
//           away from zero, computed as round(x 10^digits) / 10^digits. The hull of the rounded
//           images, which is not a superset of I
// See also: sign, Tensor::round
//
// CORA: contSet/@interval/round.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::round(int64_t digits, const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(digits >= 0, Error::wrong_value, [&] {
        return "round: the number of digits must be nonnegative, got " + std::to_string(digits);
    });
    CORA_CHECK(type == "decimals" || type == "significant", Error::wrong_value, [&] {
        return "round: the type must be 'decimals' or 'significant', got '" + type + "'";
    });
    if (type == "significant")
        CORA_FAIL(Error::not_supported, "round: the type 'significant' is not supported");
    if (digits == 0) return {inf_.round(), sup_.round()};
    const double scale = std::pow(10.0, static_cast<double>(digits));
    return {(inf_ * scale).round() / scale, (sup_ * scale).round() / scale};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
