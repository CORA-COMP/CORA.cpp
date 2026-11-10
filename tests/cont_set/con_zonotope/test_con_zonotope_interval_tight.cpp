// test_con_zonotope_interval_tight - ConZonotope::interval_tight: the box by Fourier-Motzkin passes
//
// A segment is boxed exactly by one pass; no pass gives the box of the zonotope; a set without
// points is the empty box. On random sets the box holds every point of the set, and more passes
// never give a larger box (the stated properties).
//
// CORA.cpp only: tests interval_tight, the float replacement of the programs of interval
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The segment x1 + x2 = 1 in the unit square is boxed by one pass to [0, 1]^2, none to [-1, 1]^2.
void by_hand(const std::string &b) {
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const Interval one = segment.interval_tight(1);
    check(close(one.inf(), {0, 0}, 0) && close(one.sup(), {1, 1}, 0), b + ": one pass");
    const Interval none = segment.interval_tight(0);
    check(close(none.inf(), {-1, -1}, 0) && close(none.sup(), {1, 1}, 0), b + ": no pass");
    // the constraint x1 = 3 has no solution in the box
    const ConZonotope empty(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({3.0}));
    check(empty.interval_tight(2).is_empty(), b + ": no point: the empty box");
}

/// Stated properties: the box holds the points, and k + 1 passes give a box inside that of k.
void properties(const std::string &b) {
    cora::Rng rng(31);
    for (int trial = 0; trial < 6; ++trial) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Tensor points = test::sample_points(cz, rng, 80);
        Interval previous = cz.interval_tight(0);
        for (int k = 1; k <= 6; ++k) {
            const Interval I = cz.interval_tight(k);
            check(test::inside_box(points, I), b + ": the box holds the points");
            check((previous.inf() - 1e-12).less_equal(I.inf()).min().item() > 0 &&
                      I.sup().less_equal(previous.sup() + 1e-12).min().item() > 0,
                  b + ": more passes, no larger box");
            previous = I;
        }
        const Interval exact = cz.interval();
        check(previous.inf().less_equal(exact.inf() + 1e-9).min().item() > 0 &&
                  exact.sup().less_equal(previous.sup() + 1e-9).min().item() > 0,
              b + ": not tighter than the exact box");
    }
}

/// A negative number of passes is refused; with the checks off the result is the same.
void refusals() {
    const ConZonotope cz(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { cz.interval_tight(-1); }, Error::wrong_value),
          "interval_tight: negative passes");
    const Interval on = cz.interval_tight(2);
    ChecksScope off(false);
    check(close(cz.interval_tight(2).sup(), on.sup(), 0),
          "interval_tight: the same with checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        by_hand(b);
        properties(b);
    });
    refusals();
    return test::finish("con_zonotope interval_tight");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
