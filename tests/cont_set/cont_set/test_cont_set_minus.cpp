// test_cont_set_minus - ContSet::minus: a set shifted by a vector, from either side
//
// CORA: contSet/@contSet/minus.m
// CORA authors: Mark Wetzlinger
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

/// S - p shifts by -p and p - S is -S + p, for a box and a zonotope.
void minus_shifts_the_set(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0}));
    const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.5}, {2, 2}));
    const Tensor p = column({1.0, 1.0});
    const Set A = static_cast<const ContSet &>(I).minus(p),
              B = static_cast<const ContSet &>(Z).minus(p);
    check(test::close(A.as<Interval>().inf(), {-2.0, -1.0}) &&
              test::close(A.as<Interval>().sup(), {0.0, 1.0}),
          b + ": a box minus a vector");
    check(test::close(B.as<Zonotope>().center(), {0.0, -1.0}) &&
              test::close(B.as<Zonotope>().generators(), Z.generators()),
          b + ": a zonotope minus a vector");
    const Set C = ContSet::minus(p, I), D = ContSet::minus(p, Z);
    check(test::close(C.as<Interval>().inf(), {0.0, -1.0}) &&
              test::close(C.as<Interval>().sup(), {2.0, 1.0}),
          b + ": a vector minus a box");
    check(test::close(D.as<Zonotope>().center(), {0.0, 1.0}) &&
              test::close(test::support(D, {1.0, 0.0}), 0.5) &&
              test::close(test::support(D, {-1.0, 0.0}), 0.5),
          b + ": a vector minus a zonotope");
}

/// A vector of the wrong size is refused; a class without the shift raises noops; the checks off
/// skip the check.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    const ContSet &S = I;
    check(test::raises([&] { S.minus(column({1.0})); }, Error::dimension_mismatch), "one entry");
    check(test::raises([&] { ContSet::minus(column({1.0}), S); }, Error::dimension_mismatch),
          "one entry, from the left");
    check(test::raises([] { test::UnregisteredSet().minus(column({1.0, 1.0})); }, Error::noops),
          "noops");
    ChecksScope off(false);
    check(S.minus(column({1.0, 1.0})).is<Interval>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(minus_shifts_the_set);
    errors_and_their_switch();
    return test::finish("cont_set minus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
