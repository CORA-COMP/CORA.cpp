// priv_broadcastable - whether two shapes broadcast to one, as numpy's rule
//
// Syntax:   bool ok = priv_broadcastable(a, b);
// Inputs:   a, b - shapes (lists of sizes)
// Outputs:  ok - whether, from the last axis on, each pair of sizes is equal or has a 1
// See also: rdivide, power
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_broadcastable(const std::vector<int64_t> &a, const std::vector<int64_t> &b);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
