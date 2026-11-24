// test_stl_interval_representsa - whether an interval represents a type of set
//
// CORA: specification/@stlInterval/representsa_.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl_interval.h"
#include "stl_interval_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The four types and a refused one.
void run(const std::string &b) {
    const StlInterval I(1, 3, false, true);
    check(I.representsa("stlInterval") && !I.representsa("emptySet") &&
              StlInterval().representsa("emptySet"),
          b + ": types");
    check(StlInterval(2, 2).representsa("point") && !StlInterval(1, 2).representsa("point") &&
              !StlInterval(1, 3, false, true).representsa("point"),
          b + ": point");
    check(StlInterval(0, 0).representsa("origin") && !StlInterval(0, 1).representsa("origin") &&
              !StlInterval().representsa("origin"),
          b + ": origin");
    check(test::raises([&] { I.representsa("zonotope"); }, Error::not_supported),
          b + ": another type");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(2, 2).representsa("point"), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
