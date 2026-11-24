// test_stl_minimum_time - the earliest time a formula looks at
//
// CORA: specification/@stl/minimumTime.m
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


/// The earliest time a formula looks at.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    check(p.minimum_time() == 0, b + ": non-temporal");
    check(p.finally(StlInterval(1, 2)).minimum_time() == 1, b + ": finally");
    check(p.next(1).globally(StlInterval(2, 3)).minimum_time() == 2, b + ": the outer window");
    check(p.finally(StlInterval(2, 3)).next(1).minimum_time() == 3, b + ": next below");
    check(p.until(q, StlInterval(4, 5)).minimum_time() == 0, b + ": until looks from now");
    check((p.finally(StlInterval(2, 3)) & q.next(5)).minimum_time() == 2, b + ": min of operands");
    check((~p.next(2)).minimum_time() == 2, b + ": negation");
    {
        ChecksScope off(false);
        check(p.finally(StlInterval(1, 2)).minimum_time() == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_minimum_time");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
