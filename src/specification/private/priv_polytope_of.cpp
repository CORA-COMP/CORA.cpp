// priv_polytope_of - the set of a specification as a polytope
//
// Syntax:   Polytope P = priv_polytope_of(S);
// Inputs:   S - the set
// Outputs:  P - the same set as a polytope
// See also: priv_polytope_of.h
//
// CORA.cpp only: CORA's polytope constructor dispatches on the class of a contSet
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "specification/private/priv_polytope_of.h"

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope priv_polytope_of(const Set &S) {
    // the polytope constructors of the classes that have one
    switch (S.class_id()) {
    case ClassId::polytope:
        return S.as<Polytope>();
    case ClassId::interval:
        return Polytope(S.as<Interval>());
    case ClassId::zonotope:
        return Polytope(S.as<Zonotope>());
    case ClassId::con_zonotope:
        return Polytope(S.as<ConZonotope>());
    case ClassId::zono_bundle:
        return Polytope(S.as<ZonoBundle>());
    // the classes of a curved or generated shape
    case ClassId::capsule:
        return Polytope(S.as<Capsule>());
    case ClassId::poly_zonotope:
        return Polytope(S.as<PolyZonotope>());
    case ClassId::empty_set:
        return Polytope(S.as<EmptySet>());
    case ClassId::fullspace:
        return Polytope(S.as<Fullspace>());
    default:
        CORA_FAIL(Error::noops,
                  "specification: no conversion of a " + S.class_name() + " to a polytope");
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
