// priv_check_nan - raises CORA's out-of-domain error for a result with a NaN bound
//
// See also: priv_check_nan.h
//
// CORA.cpp only: CORA's out-of-domain check of an interval result.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_check_nan.h"

#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void priv_check_nan(const std::string &name, const Tensor &lo, const Tensor &hi,
                    const std::string &domain) {
    if (lo.numel() == 0 || lo.isnan().maximum(hi.isnan()).max().item() == 0) return;
    CORA_FAIL(Error::out_of_domain, name + ": a bound is NaN (validDomain " + domain + ")");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
