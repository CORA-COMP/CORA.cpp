// test_stl_maximum_time - the latest time a formula looks at
//
// CORA: specification/@stl/maximumTime.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl.h"
#include "stl_testing.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The latest time a formula looks at.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    check(p.maximum_time() == 0, b + ": non-temporal");
    check(p.finally(StlInterval(1, 2)).maximum_time() == 2, b + ": finally");
    check(p.next(1).globally(StlInterval(0, 3)).maximum_time() == 4, b + ": next below G");
    check(p.next(1).next(2).maximum_time() == 3, b + ": next of next");
    check(p.until(q.finally(StlInterval(0, 1)), StlInterval(0, 5)).maximum_time() == 6,
          b + ": until");
    check((p.finally(StlInterval(0, 2)) & q.next(5)).maximum_time() == 5, b + ": max of operands");
    check((~p.release(q, StlInterval(1, 2))).maximum_time() == 2, b + ": negation, release");
    check(std::isinf(p.finally(StlInterval(0, INFINITY)).maximum_time()), b + ": unbounded");
    {
        ChecksScope off(false);
        check(p.finally(StlInterval(0, 2)).maximum_time() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_maximum_time");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
