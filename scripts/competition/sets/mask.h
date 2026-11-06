// mask - the answer of a containment query
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::comp {

/// One answer per point, `1` for inside. A byte per point rather than `vector<bool>`,
/// whose packed bits cannot be written from several threads at once.
using Mask = std::vector<std::uint8_t>;

} // namespace cora::comp

// ---------------------------------------  END OF CODE  ---------------------------------------- //
