// test_lean_capsule_bound - the Capsule operations that CORALean bounds, not equals
//
// supportFunc and volume (other order, rational pi); plus and radius, whose square root is Newton
// at nearest in CORALean (CORALean#274), 1 to 2 ulp from IEEE
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
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
constexpr double kSupportUlps = 4, kVolumeUlps = 512, kNormUlps = 2;

/// Whether a and b differ by at most `ulps` units in the last place of the larger.
bool aux_near(double a, double b, double ulps) {
    const double larger = std::max(std::abs(a), std::abs(b));
    const double unit = std::nextafter(larger, std::numeric_limits<double>::infinity()) - larger;
    return std::abs(a - b) <= ulps * unit;
}

/// Whether the entries of a and b are the same doubles.
bool aux_same(const Tensor &a, const Tensor &b) { return a.data() == b.data(); }

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // other summation order, rational pi: ulps apart, and below the bound in the sound mode
    oracle::run("capsule.supportFunc", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        const Tensor d = rnd.direction(3);
        const double theirs = lean::Capsule(C).support_function(d);
        test::check(aux_near(C.support_function(d).item(), theirs, kSupportUlps),
                    "capsule.supportFunc: within 4 ulp");
        if (oracle::mode() == oracle::Mode::sound)
            oracle::check_enclosed("capsule.supportFunc", C.support_function(d), theirs);
    });
    oracle::run("capsule.volume", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        const double theirs = lean::Capsule(C).volume();
        test::check(aux_near(C.volume().item(), theirs, kVolumeUlps),
                    "capsule.volume: within 512 ulp");
        if (oracle::mode() == oracle::Mode::sound)
            oracle::check_enclosed("capsule.volume", C.volume(), theirs);
    });
    // CORALean's sqrt at nearest is Newton (CORALean#274): c and g exact, r within 2 ulp
    oracle::run("capsule.plus", [](oracle::Random &rnd) {
        const Capsule a = rnd.capsule(3), b = rnd.capsule(3);
        for (const auto &[x, y] : {std::pair{a, b}, std::pair{b, a}}) {
            const Capsule mine = x.plus(y);
            if (oracle::mode() == oracle::Mode::sound) {
                oracle::check_enclosed("capsule.plus", mine, lean::Capsule(x).plus(y));
                continue;
            }
            const Capsule theirs = lean::Capsule(x).plus(y).gather();
            test::check(aux_same(mine.c(), theirs.c()) &&
                            aux_same(mine.generator(), theirs.generator()),
                        "capsule.plus: c and g exactly the oracle's");
            test::check(aux_near(mine.r().item(), theirs.r().item(), kNormUlps),
                        "capsule.plus: r within 2 ulp");
        }
    });
    // CORALean's sqrt at nearest is Newton (CORALean#274): within 2 ulp
    oracle::run("capsule.radius", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        const double theirs = lean::Capsule(C).radius();
        test::check(aux_near(C.radius().item(), theirs, kNormUlps), "capsule.radius: within 2 ulp");
        if (oracle::mode() == oracle::Mode::sound)
            oracle::check_enclosed("capsule.radius", C.radius(), theirs);
    });
    return test::finish("lean capsule bounds");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
