// priv_temporal - a node of a temporal operator over one or two formulas and a window
//
// Syntax:   Stl phi = priv_temporal(StlKind::until, a, &b, window);
// See also: priv_temporal.h
//
// CORA.cpp only: a node of a temporal operator over one or two formulas and a window
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "specification/stl/private/priv_temporal.h"
#include "specification/stl/private/priv_node.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl priv_temporal(StlKind kind, const Stl &l, const Stl *r, const StlInterval &window) {
    auto node = StlAccess::make(kind);
    node->l = StlAccess::node(l);
    if (r != nullptr) node->r = StlAccess::node(*r);
    node->interval = window;
    node->temporal = true;
    return StlAccess::wrap(node);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
