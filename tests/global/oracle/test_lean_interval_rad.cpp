// test_lean_interval_rad - Interval::rad and Interval::radius against the CORALean oracle
//
// The oracle answers radius with the squared norm of the half-widths (an upper bound of it): in
// the exact mode CORA.cpp's radius equals its square root bit for bit, in the sound mode it does
// not exceed that root by more than the rounding of the root.
//
// ORACLE: contSet/@interval/rad.m
// ORACLE: contSet/@interval/radius.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <cmath>
#include <limits>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.rad", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.rad(), lean::Interval(I).rad());
        }
    });
    test::oracle::run("interval.radius", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const double root = std::sqrt(lean::Interval(I).radius());
            if (test::oracle::mode() == test::oracle::Mode::exact)
                test::oracle::check_exact("interval.radius", I.radius(), root);
            else
                test::check(I.radius().data()[0] <=
                                root * (1 + 4 * std::numeric_limits<double>::epsilon()),
                            "interval.radius: not above the oracle's bound");
        }
    });
    return test::finish("lean interval rad");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
