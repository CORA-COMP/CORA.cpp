// test_capsule_empty - Capsule::empty: the capsule without points
//
// CORA: contSet/@capsule/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The empty capsule is flagged in its mask, holds no point, and keeps n.
void empty_known(const std::string &b) {
    const Capsule C = Capsule::empty(3);
    check(C.dim() == 3 && C.is_empty(), b + ": dimension and emptiness");
    check(C.empty_mask().all() && close(C.r(), {0}, 0) && close(C.c(), {0, 0, 0}, 0),
          b + ": flagged, the radius 0 at the origin");
    check(C.center().shape() == Shape{3} && std::isnan(C.center().data()[0]),
          b + ": the center is NaN, as the empty box's");
    check(!C.contains(test::vec({0.0, 0.0, 0.0})), b + ": it contains no point");
    check(!Capsule(vec({0.0, 0.0, 0.0})).is_empty(), b + ": a point is not empty");
    check(C.interval().is_empty(), b + ": its box is empty");
}

/// A batch is empty if every member is; a mixed batch is not.
void empty_batch(const std::string &b) {
    const Capsule none = Capsule::empty(2), point(vec({0.0, 0.0}), vec({0.0, 0.0}), 1.0);
    const Capsule all = Capsule::stack({none, none}), mixed = Capsule::stack({none, point});
    check(!mixed.is_empty(), b + ": a mixed batch is not empty");
    check(all.is_empty(), b + ": every member empty");
}

/// A negative dimension is refused; the same call with the checks off still builds an empty set.
void checks_and_their_switch() {
    check(test::raises([] { Capsule::empty(-1); }, Error::wrong_value),
          "empty: the dimension is nonnegative");
    ChecksScope off(false);
    check(Capsule::empty(2).is_empty(), "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        empty_known(b);
        empty_batch(b);
    });
    checks_and_their_switch();
    return test::finish("capsule empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
