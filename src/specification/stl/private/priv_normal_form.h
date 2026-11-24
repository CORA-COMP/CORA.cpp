// priv_normal_form - the conjunctive and disjunctive normal form by distribution
//
// Syntax:   Stl cnf = priv_normal_form(phi, StlKind::or_, StlKind::and_, StlNnf::compute);
// See also: stl.h
//
// CORA.cpp only: the conjunctive and disjunctive normal form by distribution
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Distributes `outer` over `inner` until no outer node has an inner child: | over & gives the
/// conjunctive, & over | the disjunctive normal form (CORA's passes of conjunctiveNormalForm and
/// disjunctiveNormalForm, which only swap the two operators).
Stl priv_normal_form(const Stl &phi, StlKind outer, StlKind inner, StlNnf nnf);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
