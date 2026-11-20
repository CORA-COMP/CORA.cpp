// test_cont_set_times - ContSet::times: a set scaled coordinate by coordinate, diag(f) * S
//
// CORA: contSet/@contSet/times.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
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
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A column scales per coordinate, a number scales all; for a zonotope and a box.
void times_scales(const std::string &b) {
    const Zonotope Z(vec({1.0, 2.0}), Tensor::from_data({0.5, 0.2, 0.0, 0.5}, {2, 2}));
    const Interval I(vec({-1.0, 0.0}), vec({2.0, 3.0}));
    const Tensor f = vec({2.0, -1.0});
    const Set A = static_cast<const ContSet &>(Z).times(f),
              B = static_cast<const ContSet &>(I).times(f);
    check(A.is<Zonotope>() && test::close(A.as<Zonotope>().center().expand_dims(-1), {2.0, -2.0}) &&
              test::close(A.as<Zonotope>().generators().matrix_transpose(), {1.0, 0.0, -0.2, -0.5}),
          b + ": a zonotope by a column");
    check(B.is<Interval>() && test::close(B.as<Interval>().inf(), {-2.0, -3.0}) &&
              test::close(B.as<Interval>().sup(), {4.0, 0.0}),
          b + ": a box by a column");
    const Set C = static_cast<const ContSet &>(Z).times(3.0),
              D = static_cast<const ContSet &>(I).times(-2.0);
    check(test::close(C.as<Zonotope>().center().expand_dims(-1), {3.0, 6.0}) &&
              test::close(C.as<Zonotope>().generators().matrix_transpose(), {1.5, 0.0, 0.6, 1.5}),
          b + ": a zonotope by a number");
    check(test::close(D.as<Interval>().inf(), {-4.0, -6.0}) &&
              test::close(D.as<Interval>().sup(), {2.0, 0.0}),
          b + ": a box by a number");
    // a 1 x 1 tensor is a number as well
    check(test::close(test::support(static_cast<const ContSet &>(Z).times(vec({3.0})), {1.0, 0.0}),
                      test::support(C, {1.0, 0.0})),
          b + ": a 1 x 1 factor");
    // the support function: h_{diag(f) S}(d) = h_S(f .* d)
    check(test::close(test::support(A, {1.0, 1.0}), test::support(Z, {2.0, -1.0})),
          b + ": the support of the product");
    // a zero entry times an unbounded coordinate gives 0 for a box
    const double inf = std::numeric_limits<double>::infinity();
    const Interval U(vec({-inf, 0.0}), vec({inf, 1.0}));
    const Interval R = static_cast<const ContSet &>(U).times(vec({0.0, 2.0})).as<Interval>();
    check(test::close(R.inf(), {0.0, 0.0}) && test::close(R.sup(), {0.0, 2.0}),
          b + ": 0 times unbounded");
}

/// The empty set and the whole space keep their class.
void times_of_the_extremes() {
    const Set E = static_cast<const ContSet &>(EmptySet(2)).times(vec({2.0, 3.0}));
    const Set F = static_cast<const ContSet &>(Fullspace(2)).times(vec({2.0, 3.0}));
    check(E.is<EmptySet>() && E->dim() == 2, "the empty set");
    check(F.is<Fullspace>() && F->dim() == 2, "the whole space");
}

/// A matrix factor is not supported, a column of the wrong size is refused, a class without the
/// linear map raises noops; the checks off skip the checks.
void errors_and_their_switch() {
    const Interval I(vec({0.0, 0.0}), vec({1.0, 1.0}));
    const ContSet &S = I;
    // a factor (2, 2) is a batch of two factors (#163), which a box broadcasts
    check(S.times(Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2})).as<Interval>().inf().shape() ==
              Shape({2, 2}),
          "a batch of factors");
    check(test::raises([&] { S.times(vec({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "three entries");
    check(test::raises([] { test::UnregisteredSet().times(2.0); }, Error::noops), "noops");
    ChecksScope off(false);
    check(S.times(vec({1.0, 2.0})).is<Interval>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(times_scales);
    times_of_the_extremes();
    errors_and_their_switch();
    return test::finish("cont_set times");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
