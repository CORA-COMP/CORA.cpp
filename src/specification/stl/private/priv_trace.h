// priv_trace - a sampled trace of states read against the variables of a formula
//
// Syntax:   StlTrace tr = priv_trace(phi, x, t, "robustness");
//           double v = priv_eval(e, tr.names, tr.row(i));
// Inputs:   phi - the formula
//           x - the states (m, n), one per row
//           t - the m times
//           op - the name of the operation, for messages
// Outputs:  tr - the checked trace with the names of its columns
//           v - the value of e at a state
// See also: stl.h
//
// CORA.cpp only: CORA reads the columns of x through contains(vars, name)
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/specification/stl.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A trace on the host: m states of n components at m times, and the variable of each column.
struct StlTrace {
    std::size_t m = 0, n = 0;
    std::vector<double> x, t; ///< (m, n) row-major, (m)
    std::vector<std::string> names;
    const double *row(std::size_t i) const { return x.data() + i * n; }
};

/// The variable names of the columns: those of the formula, the numbers of one stem without gaps.
std::vector<std::string> priv_trace_names(const Stl &phi);

/// The trace of x and t read for phi; Error::wrong_value if the sizes do not fit the formula.
StlTrace priv_trace(const Stl &phi, const Tensor &x, const std::vector<double> &t, const char *op);

/// The value of the expression at a state, the variables read by their column in names.
double priv_eval(const StlExpr &e, const std::vector<std::string> &names, const double *state);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
