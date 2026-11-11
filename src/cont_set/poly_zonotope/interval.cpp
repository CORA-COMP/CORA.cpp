// interval - the box around a polynomial zonotope, as CORA's polyZonotope.interval
//
// The box of the enclosing zonotope (method "interval"); the bounds of the other methods
// (split, bnb, bernstein) are not ported.
//
// Syntax:   I = pZ.interval();
// Inputs:   -
// Outputs:  I - the box (n, 1)
// See also: zonotope
//
// CORA: contSet/@polyZonotope/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval PolyZonotope::interval_impl() const { return zonotope().interval(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
