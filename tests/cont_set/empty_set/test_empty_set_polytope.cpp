// test_empty_set_polytope - Polytope(EmptySet): the empty set as a polytope
//
// CORA: contSet/@emptySet/polytope.m
//
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/polytope/polytope.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The polytope of the empty set of n dimensions is empty and has dimension n.
void conversions(const std::string &b) {
    for (const int64_t n : {int64_t{0}, int64_t{1}, int64_t{3}}) {
        const Polytope P{EmptySet(n)};
        check(P.dim() == n && P.is_h_rep(), b + ": the dimension " + std::to_string(n));
        check(n == 0 || P.representsa("emptySet"), b + ": empty, dimension " + std::to_string(n));
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(conversions);
    // the input is an empty set: a negative dimension is refused
    check(test::raises([] { EmptySet(-1); }, Error::wrong_input_in_constructor),
          "EmptySet: n >= 0");
    ChecksScope off(false);
    check(Polytope(EmptySet(2)).dim() == 2, "Polytope(EmptySet): the same with the checks off");
    return test::finish("empty_set polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
