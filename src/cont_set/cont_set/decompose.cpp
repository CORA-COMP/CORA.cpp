// decompose - a set cut into blocks of coordinates, as CORA's contSet.decompose
//
// CORA returns the unprojected set for a single block (docs/cora-differences.md); CORA.cpp
// projects.
//
// Syntax:   Sub = S.decompose({{0, 1}, {2, 4}});
// Inputs:   S - set of dimension n
//           blocks - first and last coordinate of each block (0-based, inclusive) in 0..n-1
// Outputs:  Sub - the projection of S onto each block, in the order of blocks
// See also: project
//
// CORA: contSet/@contSet/decompose.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Set> ContSet::decompose(const std::vector<std::array<int64_t, 2>> &blocks) const {
    detail::Entry entry;
    CORA_CHECK(!blocks.empty(), Error::wrong_value,
               [] { return std::string("decompose: name at least one block, e.g. {{0, 1}}"); });
    for (const std::array<int64_t, 2> &block : blocks)
        CORA_CHECK(0 <= block[0] && block[0] <= block[1] && block[1] < dim(), Error::wrong_value,
                   [&] {
                       return "decompose: the block {" + std::to_string(block[0]) + ", " +
                              std::to_string(block[1]) + "} must satisfy 0 <= first <= last < " +
                              std::to_string(dim());
                   });
    std::vector<Set> parts;
    for (const std::array<int64_t, 2> &block : blocks) {
        std::vector<int64_t> dims;
        for (int64_t d = block[0]; d <= block[1]; ++d) dims.push_back(d);
        parts.push_back(project(dims));
    }
    return parts;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
