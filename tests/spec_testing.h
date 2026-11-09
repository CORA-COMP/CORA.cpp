// spec_testing - halfspaces and polytopes for the tests of specifications
//
// A specification takes a set, typically a Polytope. The tests state their sets as the
// halfspaces {x | a'x <= b} of the old factories: `halfspace(a, b)` is one such polytope and
// `polytope({{a1, b1}, {a2, b2}})` stacks several rows into one.
//
// Syntax:   Specification::safe_set(test::halfspace(test::column({1, 0}), 3.0));
// See also: testing.h, cora/specification/specification.h
//
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/polytope/polytope.h"
#include "testing.h"

#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The halfspace {x | a'x <= b} of the column a (n, 1) as a polytope.
inline cora::Polytope halfspace(const cora::Tensor &a, double b) {
    return cora::Polytope(a.matrix_transpose(), a.full_like({1, 1}, b));
}

/// The intersection of halfspaces (a, b), one row each.
inline cora::Polytope polytope(const std::vector<std::pair<cora::Tensor, double>> &rows) {
    std::vector<cora::Tensor> A, b;
    for (const auto &[a, offset] : rows) {
        A.push_back(a.matrix_transpose());
        b.push_back(a.full_like({1, 1}, offset));
    }
    return cora::Polytope(cora::Tensor::concat(A, -2), cora::Tensor::concat(b, -2));
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
