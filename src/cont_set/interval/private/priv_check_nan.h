// priv_check_nan - raises CORA's out-of-domain error for a result with a NaN bound
//
// Syntax:   priv_check_nan("power", lo, hi, "base >= 0");
// Inputs:   name - the operation, for the message;  lo, hi - the bounds of the result;
//           domain - the valid domain, for the message
// Outputs:  none; Error::out_of_domain if some bound is NaN, as CORA's check at the end of power
//           and rdivide
// See also: priv_check_domain

#pragma once

#include "cora/global/tensor/tensor.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

void priv_check_nan(const std::string &name, const Tensor &lo, const Tensor &hi,
                    const std::string &domain);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
