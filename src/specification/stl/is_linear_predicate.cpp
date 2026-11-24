// is_linear_predicate - whether a formula is a conjunction of linear comparisons
//
// Syntax:   bool b = phi.is_linear_predicate();
// Inputs:   none
// Outputs:  b - whether it is built from comparisons of linear expressions and & only
// See also: stl.h
//
// CORA: specification/@stl/isLinearPredicate.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Stl::is_linear_predicate() const {
    detail::Entry entry;
    if (kind() == StlKind::comparison) return expr_lhs().is_linear() && expr_rhs().is_linear();
    return kind() == StlKind::and_ && lhs().is_linear_predicate() && rhs().is_linear_predicate();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
