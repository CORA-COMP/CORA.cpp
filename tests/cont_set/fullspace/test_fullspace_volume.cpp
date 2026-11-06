// test_fullspace_volume - fullspace volume: Inf
//
// CORA: contSet/@fullspace/volume_.m
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

void the_volume_is_infinite(const std::string &b) {
    for (const int64_t n : {1, 2, 7})
        check(Fullspace(n).volume() == std::numeric_limits<double>::infinity(),
              b + ": dimension " + std::to_string(n));
    check(std::isnan(Fullspace(0).volume()), b + ": NaN in R^0, as CORA");
}

/// volume takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(std::isinf(Fullspace(2).volume()), "volume: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_volume_is_infinite);
    checks_and_their_switch();
    return test::finish("fullspace volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
