// test_lean_capsule_bound - the Capsule operations that CORALean bounds, not equals
//
// volume (rational pi)
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/oracle/capsule.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// How far CORA.cpp may be from the oracle, in units in the last place.
constexpr double kVolumeUlps = 512;

/// Whether a and b differ by at most `ulps` units in the last place of the larger.
bool aux_near(double a, double b, double ulps) {
    const double larger = std::max(std::abs(a), std::abs(b));
    const double unit = std::nextafter(larger, std::numeric_limits<double>::infinity()) - larger;
    return std::abs(a - b) <= ulps * unit;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // rational pi: ulps apart
    oracle::run("capsule.volume", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        const double theirs = lean::Capsule(C).volume();
        test::check(aux_near(C.volume().item(), theirs, kVolumeUlps),
                    "capsule.volume: within 512 ulp");
    });
    return test::finish("lean capsule bounds");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
