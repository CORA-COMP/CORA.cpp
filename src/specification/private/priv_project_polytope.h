// priv_project_polytope - the point of a polytope nearest to a point
//
// Syntax:   bool ok = priv_project_polytope(A, b, n, p, q);
// Inputs:   A - the rows (m, n) row-major of {x | A x <= b}
//           b - the right-hand sides (m)
//           n - the dimension
//           p - the point (n)
// Outputs:  ok - false if the polytope is empty
//           q - the nearest point (n)
// See also: Polytope
//
// CORA.cpp only: stands in for CORAquadprog with H = I and f = -p, as robustness asks it
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstddef>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_project_polytope(const std::vector<double> &A, const std::vector<double> &b,
                           std::size_t n, const std::vector<double> &p, std::vector<double> &q);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
