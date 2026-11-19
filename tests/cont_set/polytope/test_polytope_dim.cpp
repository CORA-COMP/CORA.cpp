// test_polytope_dim - Polytope::dim: the dimension of either form
//
// CORA: contSet/@polytope/dim.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dimension of the halfspace form, the vertex form, R^n, the empty set and a batch.
void dimensions(const std::string &b) {
    check(test::square_h().dim() == 2, b + ": halfspaces");
    check(Polytope(Tensor::ones({5, 3})).dim() == 3, b + ": vertices");
    check(Polytope::inf(4).dim() == 4 && Polytope::empty(2).dim() == 2, b + ": R^n and empty");
    // a batch has the dimension of its members
    check(Polytope(Tensor::ones({3, 2, 3}), Tensor::ones({3, 2})).dim() == 3, b + ": a batch");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(dimensions);
    // with the checks off
    ChecksScope off(false);
    check(test::square_v().dim() == 2, "dim: the same with the checks off");
    return test::finish("polytope dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
