// validate_step - the checks that every step operation of linear_sys shares
//
// Syntax:   if (entry.check()) validate_step("one_step", X, time_step, taylor_terms);
// Inputs:   op - name of the operation, for the message
//           X - set of the step
//           time_step - step size
//           taylor_terms - order of the series
// Outputs:  none; an Error if X is not in the state space, time_step <= 0 or taylor_terms < 1
// See also: validate, one_step, homogeneous_solution
//
// CORA.cpp only: CORA validates through its options structs (validateOptions).
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void LinearSys::validate_step(const char *op, const Zonotope &X, double time_step,
                              int taylor_terms) const {
    detail::Validation entry;
    // the set lives in the state space, the step has a size and the series has terms
    CORA_CHECK(X.dim() == dim(), Error::dimension_mismatch, [&] {
        return std::string(op) + ": the set must have the dimension " + std::to_string(dim()) +
               " of the states, got " + std::to_string(X.dim());
    });
    CORA_CHECK(time_step > 0 && taylor_terms >= 1, Error::wrong_value, [&] {
        return std::string(op) + ": needs time_step > 0 and taylor_terms >= 1, got " +
               std::to_string(time_step) + " and " + std::to_string(taylor_terms);
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
