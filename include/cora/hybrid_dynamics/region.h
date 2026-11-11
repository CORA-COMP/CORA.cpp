// region - the set an invariant or a guard of a hybrid automaton is
//
// CORA lets the invariant and the guard of a location be a polytope, a level set or (guards) an
// interval or fullspace. A Region is a polytope (a box is held as the polytope of its sides) or a
// level set. Every code path that worked on the polytope still does, unchanged; an operation
// that is not ported for level sets refuses them with Error::not_supported, naming itself.
//
// Syntax:     Region r = polytope;   Region r = LevelSet(f, n, CompOp::eq);
//             const Polytope &P = region_polytope(r, "potInt");
// Operations: region_dim, region_polytope, region_level_set
// See also:   cora/hybrid_dynamics/location/location.h, transition.h
//
// CORA.cpp only: CORA's invariant and guard are any contSet; here the classes a location handles
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/level_set/level_set.h"
#include "cora/cont_set/polytope/polytope.h"

#include <variant>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

using Region = std::variant<Polytope, LevelSet>;

/// The dimension of the region.
inline int64_t region_dim(const Region &r) {
    return std::visit([](const auto &s) { return s.dim(); }, r);
}

/// The polytope of a region; a level set is refused with Error::not_supported naming `what`.
inline const Polytope &region_polytope(const Region &r, const char *what) {
    if (const auto *P = std::get_if<Polytope>(&r)) return *P;
    CORA_FAIL(Error::not_supported,
              std::string(what) + ": a level set invariant or guard is not supported here");
}

/// The level set of a region, or null for a polytope.
inline const LevelSet *region_level_set(const Region &r) { return std::get_if<LevelSet>(&r); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
