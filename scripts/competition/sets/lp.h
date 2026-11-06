// lp - the exact containment query, as a linear program
//
// `r` is in `{G b | ||b||_inf <= 1}` iff `min {t | G b = r, -t <= b <= t}` is at most one.
// This settles the points the projections leave open and the flat zonotopes the facets
// cannot describe; the catalog's instances reach it rarely, so it is a general-purpose
// simplex rather than anything tuned.
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <Eigen/Dense>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::comp {

/// Whether `G b = r` has a solution with `||b||_inf <= 1 + tol`.
bool contains_lp(const Eigen::Ref<const Eigen::MatrixXd> &g,
                 const Eigen::Ref<const Eigen::VectorXd> &r, double tol);

} // namespace cora::comp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
