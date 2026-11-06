// test_fullspace_polytope - Polytope(Fullspace): the whole space as a polytope
//
// CORA: contSet/@fullspace/polytope.m
//
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/polytope/polytope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The polytope of R^n has no rows and holds every point.
void conversions(const std::string &b) {
    for (const int64_t n : {int64_t{1}, int64_t{3}}) {
        const Polytope P{Fullspace(n)};
        check(P.dim() == n && P.A().shape()[0] == 0 && P.representsa("fullspace"),
              b + ": R^" + std::to_string(n));
    }
    check(Polytope(Fullspace(2)).contains(Tensor::ones({2, 1}) * 1e9), b + ": every point");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(conversions);
    // the input is the whole space: a negative dimension is refused
    check(test::raises([] { Fullspace(-1); }, Error::wrong_input_in_constructor),
          "Fullspace: n >= 0");
    ChecksScope off(false);
    check(Polytope(Fullspace(2)).dim() == 2, "Polytope(Fullspace): the same with the checks off");
    return test::finish("fullspace polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
