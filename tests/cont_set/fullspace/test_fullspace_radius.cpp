// test_fullspace_radius - fullspace radius: Inf
//
// CORA: contSet/@fullspace/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_radius_is_infinite(const std::string &b) {
    for (const int64_t n : {1, 2, 7})
        check(Fullspace(n).radius() == std::numeric_limits<double>::infinity(),
              b + ": dimension " + std::to_string(n));
    check(std::isnan(Fullspace(0).radius()), b + ": NaN in R^0, as CORA");
}

/// radius takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(std::isinf(Fullspace(2).radius()), "radius: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_radius_is_infinite);
    checks_and_their_switch();
    return test::finish("fullspace radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
