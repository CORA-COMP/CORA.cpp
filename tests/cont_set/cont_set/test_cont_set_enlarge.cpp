// test_cont_set_enlarge - ContSet::enlarge: a set scaled about its center, in the class of the set
//
// CORA: contSet/@contSet/enlarge.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A box grows by its own formula, a zonotope by the generic one: the center stays and the
/// generators scale by the factor of their coordinate.
void enlarge_scales_about_the_center(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0}));
    const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    const ContSet &A = I, &B = Z;
    const Set RI = A.enlarge(2.0), RJ = A.enlarge(column({2.0, 0.5}));
    check(test::close(RI.as<Interval>().inf(), {-2.0, -1.0}) &&
              test::close(RI.as<Interval>().sup(), {2.0, 3.0}),
          b + ": a box doubles");
    check(test::close(RJ.as<Interval>().inf(), {-2.0, 0.5}) &&
              test::close(RJ.as<Interval>().sup(), {2.0, 1.5}),
          b + ": a factor per coordinate");
    const Set RZ = B.enlarge(2.0), RW = B.enlarge(column({2.0, 3.0}));
    check(RZ.is<Zonotope>() && test::close(RZ.as<Zonotope>().center(), {1.0, 0.0}) &&
              test::close(RZ.as<Zonotope>().generators(), {1.0, 0.0, 0.0, 1.0}),
          b + ": a zonotope doubles about its center");
    check(test::close(RW.as<Zonotope>().generators(), {1.0, 0.0, 0.0, 1.5}),
          b + ": a factor per coordinate of a zonotope");
    check(test::close(test::support(B.enlarge(0.0), {1.0, 0.0}), 1.0),
          b + ": factor 0 leaves the center");
}

/// A factor of the wrong size is refused; a class without the shift raises noops; the checks off
/// skip the check.
void errors_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const ContSet &S = Z;
    check(test::raises([&] { S.enlarge(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "three factors");
    check(test::raises([] { test::UnregisteredSet().enlarge(2.0); }, Error::noops), "noops");
    ChecksScope off(false);
    check(S.enlarge(column({1.0, 2.0})).is<Zonotope>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(enlarge_scales_about_the_center);
    errors_and_their_switch();
    return test::finish("cont_set enlarge");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
