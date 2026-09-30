// priv - the private functions of specification, as CORA's @specification/private
//
// Not for callers of the class: check uses them.
//
// Syntax:   bool hit = priv_zonotopeMeetsPolytope(c, G, m, a, b);
// See also: specification.h, check.cpp

#pragma once

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether the zonotope {c + G beta | |beta|_inf <= 1} meets the polytope {x | a[i]'x <= b[i]}:
/// a linear program in beta, decided by a phase-1 simplex. c has n values, G is n x m row-major,
/// a[i] has n values. Touching counts as meeting, and a tolerance of 1e-9 on the infeasibility
/// errs towards "meets", so a caller that treats "meets" as a violation is never wrongly safe.
bool priv_zonotopeMeetsPolytope(const std::vector<double> &c, const std::vector<double> &G, int m,
                                const std::vector<std::vector<double>> &a,
                                const std::vector<double> &b);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
