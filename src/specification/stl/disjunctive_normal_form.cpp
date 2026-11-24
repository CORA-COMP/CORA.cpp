// disjunctive_normal_form - the disjunctive normal form of a formula
//
// Syntax:   res = phi.disjunctive_normal_form(nnf);
// Inputs:   nnf - StlNnf::given if phi is in negation normal form already
// Outputs:  res - a disjunction of conjunctions over the non-logic parts
// See also: stl.h
//
// CORA: specification/@stl/disjunctiveNormalForm.m
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

Stl Stl::disjunctive_normal_form(StlNnf nnf) const {
    detail::Entry entry;
    return priv_normal_form(*this, StlKind::and_, StlKind::or_, nnf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
