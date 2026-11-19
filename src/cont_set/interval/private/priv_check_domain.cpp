// priv_check_domain - raises CORA's out-of-domain error for a box beyond a function's domain
//
// Syntax:   priv_check_domain("acos", I, -1, 1);
// Inputs:   name - the function, for the message
//           I - the box (its empty members are not checked)
//           low, high - the domain [low, high] (high may be +Inf)
// Outputs:  -   (Error::out_of_domain if some bound lies outside, as CORA's NaN check does)
// See also: acos, acosh
//
// CORA.cpp only: CORA's out-of-domain check of an interval function.
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_check_domain.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void priv_check_domain(const std::string &name, const Interval &I, double low, double high) {
    if (I.is_empty()) return;
    // a NaN bound fails both comparisons, as it makes CORA's result NaN
    bool inside = true;
    if (!I.empty_mask().any()) {
        inside = I.inf().min().item() >= low && I.sup().max().item() <= high;
    } else {
        const std::vector<uint8_t> empty = I.empty_mask().flags(I.batch_shape());
        const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
        const std::size_t per = lo.size() / empty.size(); // the entries of one member
        for (std::size_t k = 0; k < lo.size(); ++k)
            if (!empty[k / per] && !(lo[k] >= low && hi[k] <= high)) inside = false;
    }
    if (inside) return;
    CORA_FAIL(Error::out_of_domain, name + ": the bounds must lie in [" + std::to_string(low) +
                                        ", " + std::to_string(high) + "] (validDomain)");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
