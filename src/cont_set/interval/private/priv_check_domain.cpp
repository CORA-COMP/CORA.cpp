// priv_check_domain - raises CORA's out-of-domain error for a box beyond a function's domain
//
// Syntax:   priv_check_domain("acos", inf, sup, -1, 1);
// Inputs:   name - the function, for the message;  inf, sup - the bounds of the box;
//           low, high - the domain [low, high] (high may be +Inf)
// Outputs:  none; Error::out_of_domain if some bound lies outside, as CORA's NaN check does
// See also: acos, acosh

#include "cont_set/interval/private/priv_check_domain.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void priv_check_domain(const std::string &name, const Tensor &inf, const Tensor &sup, double low,
                       double high) {
    // a NaN bound fails both comparisons, as it makes CORA's result NaN
    if (inf.numel() == 0 || (inf.min().item() >= low && sup.max().item() <= high)) return;
    CORA_FAIL(Error::out_of_domain, name + ": the bounds must lie in [" + std::to_string(low) +
                                        ", " + std::to_string(high) + "] (validDomain)");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
