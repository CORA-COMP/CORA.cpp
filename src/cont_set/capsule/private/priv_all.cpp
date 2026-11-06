// priv_all - whether every entry of a 0/1 mask is 1
//
// Syntax:   bool all = priv_all(mask);
// Inputs:   mask - a 0/1 tensor, as the comparisons of Tensor return
// Outputs:  all - true if no entry is 0, also for a mask without entries
// See also: priv_at_most
//
// CORA.cpp only: CORA's predicates return one number per set; the batch answers all of them
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_all(const Tensor &mask) { return mask.numel() == 0 || mask.detach().min().item() == 1.0; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
