// specification - the oracle's check of a specification on a box or a zonotope (CORALean)
//
// Syntax:     bool proved = lean::check_specification("safeSet", A, b, I);
// Meaning:    the specification {x | A x <= b} of kind "safeSet", "unsafeSet" or "invariant"
//             against a box or a zonotope: true proves the set contained (safeSet, invariant) or
//             disjoint from the specification's set (unsafeSet); false says only that no proof was
//             found
// See also:   lean/request.h, cora/specification/specification.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/zonotope.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// Whether the oracle proves that the box / zonotope satisfies the specification of the polytope
/// A x <= b (A is (p, n), b is (p, 1)).
bool check_specification(const std::string &kind, const Tensor &A, const Tensor &b,
                         const Interval &I);
bool check_specification(const std::string &kind, const Tensor &A, const Tensor &b,
                         const Zonotope &Z);

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
