// test_capsule_center - Capsule center: the middle of the axis
//
// CORA: contSet/@capsule/center.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The center is the stored c, for a segment, a ball and a batch; the empty capsule has none.
void center_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(close(C.center(), {1, 2}, 0), b + ": the stored center");
    check(close(Capsule(column({5.0})).center(), {5}, 0), b + ": a point is its own center");
    check(Capsule::empty(2).center().shape() == Shape{2, 0}, b + ": empty: no column");
    const Tensor c = Tensor::from_data({1, 2, 3, 4}, {2, 2, 1});
    check(close(Capsule(c, c, 0.1).center(), c, 0), b + ": a batch gives the stored centers");
    check(Capsule(column({1.0, 2.0})).center().shape() == Shape{2, 1}, b + ": a column");
}

/// The center is a point of the capsule: random capsules contain their centers.
void center_property(const std::string &b) {
    Rng rng(11);
    for (int k = 0; k < 8; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        check(C.contains(C.center()), b + ": the center is in the capsule");
    }
}

/// With the checks off a valid call gives the same center.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    const Tensor on = C.center();
    ChecksScope off(false);
    check(close(C.center(), on, 0), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        center_known(b);
        center_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
