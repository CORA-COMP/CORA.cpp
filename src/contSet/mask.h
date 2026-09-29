// The answer of a containment query.

#pragma once

#include <cstdint>
#include <vector>

namespace cora {

/// One answer per point, `1` for inside. A byte per point rather than `vector<bool>`,
/// whose packed bits cannot be written from several threads at once.
using Mask = std::vector<std::uint8_t>;

} // namespace cora
