// priv - the private functions of specification, as CORA's @specification/private
//
// Not for callers of the class: check uses them.
//
// Syntax:   bool hit = priv_zonotope_meets_polytope(c, G, m, a, b);
// See also: specification.h, check.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       30-September-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether the zonotope {c + G beta | |beta|_inf <= 1} meets the polytope {x | a[i]'x <= b[i]}: a
/// linear program in beta by a phase-1 simplex; touching counts as meeting.
bool priv_zonotope_meets_polytope(const std::vector<double> &c, const std::vector<double> &G, int m,
                                  const std::vector<std::vector<double>> &a,
                                  const std::vector<double> &b);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
