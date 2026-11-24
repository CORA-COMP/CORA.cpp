// test_stl_is_linear_predicate - whether a formula is a conjunction of linear comparisons
//
// CORA: specification/@stl/isLinearPredicate.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// Whether a formula is a conjunction of linear comparisons.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl lin = (2.0 * x[0] + x[1]) <= 3.0;
    check(lin.is_linear_predicate() && (lin & p).is_linear_predicate(), b + ": linear");
    check(!(x[0] * x[1] < 1.0).is_linear_predicate(), b + ": product of variables");
    check(!(x[0].power(2) < 1.0).is_linear_predicate(), b + ": power");
    check((x[0] * 3.0 - x[1] > 0.0).is_linear_predicate(), b + ": scaled");
    check(!(p | q).is_linear_predicate() && !p.next(1).is_linear_predicate(),
          b + ": not a conjunction");
    {
        ChecksScope off(false);
        check(lin.is_linear_predicate(), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_is_linear_predicate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
