// test_stl_next - the next operator and its time
//
// CORA: specification/@stl/next.m
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


/// The next operator and its time.
void run(const std::string &b) {
    const auto x = StlExpr::variables("x", 2);
    const Stl p = x[0] < 5.0, q = x[1] >= 0.0;

    const Stl phi = p.next(1.5);
    check(phi.kind() == StlKind::next && phi.from() == 1.5 && phi.to() == 1.5 && phi.temporal(),
          b + ": next");
    check(!p.temporal() && p.next(0).lhs().isequal(p), b + ": the operand");
    // a negative or infinite time is refused
    check(test::raises([&] { p.next(-1); }, Error::wrong_input_in_constructor), b + ": negative");
    check(test::raises([&] { p.next(INFINITY); }, Error::wrong_input_in_constructor),
          b + ": infinite");
    {
        ChecksScope off(false);
        check(p.next(2).from() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_next");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
