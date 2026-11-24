// test_stl_interval_to_string - the text of an interval
//
// CORA: specification/@stlInterval/display.m
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


/// The text of closed, half open and empty intervals.
void run(const std::string &b) {
    check(StlInterval(0, 2, true, false).to_string() == "[0, 2)", b + ": half open");
    check(StlInterval(1, 2, false, true).to_string() == "(1, 2]", b + ": open left");
    check(StlInterval().to_string() == "{}", b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).to_string() == "[1, 2]", b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_to_string");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
