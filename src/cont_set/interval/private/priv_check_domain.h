// priv_check_domain - raises CORA's out-of-domain error for a box beyond a function's domain
//
// Syntax:   priv_check_domain("acos", inf, sup, -1, 1);
// Inputs:   name - the function, for the message
//           inf, sup - the bounds of the box;
//           low, high - the domain [low, high] (high may be +Inf)
// Outputs:  -   (Error::out_of_domain if some bound lies outside, as CORA's NaN check does)
// See also: acos, acosh
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

void priv_check_domain(const std::string &name, const Tensor &inf, const Tensor &sup, double low,
                       double high);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
