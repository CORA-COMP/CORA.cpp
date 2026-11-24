// is_predicate - whether a formula is a conjunction of comparisons
//
// Syntax:   bool b = phi.is_predicate();
// Inputs:   none
// Outputs:  b - whether it is built from comparisons and & only
// See also: stl.h
//
// CORA: specification/@stl/isPredicate.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Stl::is_predicate() const {
    detail::Entry entry;
    if (kind() == StlKind::comparison) return true;
    return kind() == StlKind::and_ && lhs().is_predicate() && rhs().is_predicate();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
