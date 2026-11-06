// validate - the checks of a reach problem against a linear system
//
// reach runs them once when it checks its input; an algorithm registered from Python runs the
// same, so a prototype gets the messages of the built-in algorithms.
//
// Syntax:   sys.validate(problem);
// Inputs:   problem - initial_set, input_set (none, or a zonotope of the columns of B), t_final
// Outputs:  -  (an Error if the problem does not fit the system)
// See also: reach
//
// CORA.cpp only: MATLAB CORA checks params inside each reach (validateOptions).
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void LinearSys::validate(const ReachProblem &problem) const {
    detail::Validation entry;
    // the initial set of the system's dimension and a positive, finite horizon
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(problem.initial_set->dim() == dim(), Error::dimension_mismatch, [&] {
        return "reach: the initial set has the dimension " +
               std::to_string(problem.initial_set->dim()) + ", the system " + std::to_string(dim());
    });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (!problem.input_set) return;
    // an input set: B, the dimension of its columns, and no batch
    const Zonotope &U = *problem.input_set;
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("reach: an input_set needs B; use LinearSys(A, B)"); });
    CORA_CHECK(U.dim() == B_->shape().back(), Error::dimension_mismatch, [&] {
        return "reach: the input set needs the dimension " + std::to_string(B_->shape().back()) +
               " of the columns of B, got " + std::to_string(U.dim());
    });
    CORA_CHECK(A_.shape().size() == 2 && problem.initial_set->center().shape().size() == 2,
               Error::not_supported, [] {
                   return std::string("reach: with an input set, one system and one initial set");
               });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
