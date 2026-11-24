// conjunctive_normal_form - the conjunctive normal form of a formula
//
// Syntax:   res = phi.conjunctive_normal_form(nnf);
// Inputs:   nnf - StlNnf::given if phi is in negation normal form already
// Outputs:  res - a conjunction of disjunctions over the non-logic parts
// See also: stl.h
//
// CORA: specification/@stl/conjunctiveNormalForm.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

#include "specification/stl/private/priv_normal_form.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::conjunctive_normal_form(StlNnf nnf) const {
    detail::Entry entry;
    return priv_normal_form(*this, StlKind::or_, StlKind::and_, nnf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
