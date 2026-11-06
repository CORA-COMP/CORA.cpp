// test_cont_set_uplus - ContSet::uplus: the set itself, with every number as it was
//
// CORA: contSet/@contSet/uplus.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// +S equals S, down to the sign of a zero, for a zonotope and a box.
void uplus_is_the_identity(const std::string &b) {
    const Zonotope Z(column({-0.0, 1.0}), Tensor::from_data({0.5, 0.0, -0.0, 0.5}, {2, 2}));
    const Interval I(column({-0.0, 0.0}), column({2.0, 3.0}));
    const Set A = static_cast<const ContSet &>(Z).uplus(),
              B = static_cast<const ContSet &>(I).uplus();
    check(A.is<Zonotope>() && test::close(A.as<Zonotope>().center(), Z.center(), 0) &&
              test::close(A.as<Zonotope>().generators(), Z.generators(), 0),
          b + ": a zonotope");
    check(std::signbit(A.as<Zonotope>().center().slice(-2, 0, 1).item()), b + ": the sign of -0");
    check(B.is<Interval>() && test::close(B.as<Interval>().inf(), I.inf(), 0) &&
              test::close(B.as<Interval>().sup(), I.sup(), 0),
          b + ": a box");
}

/// The empty set and the whole space stay what they are.
void uplus_of_the_extremes() {
    check(static_cast<const ContSet &>(EmptySet(2)).uplus().is<EmptySet>(), "the empty set");
    check(static_cast<const ContSet &>(Fullspace(2)).uplus().is<Fullspace>(), "the whole space");
}

/// A class without the translation raises noops; the checks off change nothing.
void errors_and_their_switch() {
    check(test::raises([] { test::UnregisteredSet().uplus(); }, Error::noops), "noops");
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    ChecksScope off(false);
    check(static_cast<const ContSet &>(I).uplus().is<Interval>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(uplus_is_the_identity);
    uplus_of_the_extremes();
    errors_and_their_switch();
    return test::finish("cont_set uplus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
