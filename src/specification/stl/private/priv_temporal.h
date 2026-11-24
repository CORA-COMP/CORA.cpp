// priv_temporal - a node of a temporal operator over one or two formulas and a window
//
// Syntax:   Stl phi = priv_temporal(StlKind::until, a, &b, window);
// See also: stl.h
//
// CORA.cpp only: a node of a temporal operator over one or two formulas and a window
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The formula of a temporal kind (until, release, finally, globally, next) over l (and r for
/// until and release) with the window.
Stl priv_temporal(StlKind kind, const Stl &l, const Stl *r, const StlInterval &window);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
