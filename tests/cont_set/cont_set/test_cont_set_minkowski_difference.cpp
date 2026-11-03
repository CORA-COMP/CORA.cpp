// test_cont_set_minkowski_difference - ContSet::minkowski_difference: {x | x + S2 in S}
//
// CORA: contSet/@contSet/minkDiff.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// A box minus a box subtracts the bounds; a zonotope minus a box or a zonotope keeps the sum
/// inside the minuend.
void the_difference_of_any_pair(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})),
        S(column({-0.5, 0.0}), column({0.5, 0.5}));
    const Set R = static_cast<const ContSet &>(I).minkowski_difference(S);
    check(test::close(R.as<Interval>().inf(), {-0.5, 0.0}) &&
              test::close(R.as<Interval>().sup(), {0.5, 1.5}),
          b + ": a box minus a box");
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const Zonotope Q(column({0.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.0, 0.25}, {2, 2}));
    const Interval box(column({-0.5, -0.25}), column({0.5, 0.25}));
    for (const ContSet *T : {static_cast<const ContSet *>(&Q), static_cast<const ContSet *>(&box)}) {
        const Set D = static_cast<const ContSet &>(Z).minkowski_difference(*T, "inner");
        check(D.is<Zonotope>(), b + ": a zonotope minus a zonotope or a box");
        for (const std::vector<double> &d : {std::vector<double>{1, 0}, {0, 1}, {1, 1}, {1, -1}})
            check(test::support(D, d) + test::support(*T, d) <= test::support(Z, d) + 1e-9,
                  b + ": the difference plus the subtrahend lies in the minuend");
        check(test::close(test::support(D, {1.0, 0.0}), 0.5), b + ": the exact width in x");
    }
}

/// A set of another dimension is refused; a pair without an algorithm raises noops; the checks off
/// skip the dimension check.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0})), I3(column({0.0}), column({1.0}));
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    check(test::raises([&] { static_cast<const ContSet &>(I).minkowski_difference(I3); },
                       Error::dimension_mismatch),
          "dimensions 2 and 1");
    check(test::raises([&] { static_cast<const ContSet &>(Z).minkowski_difference(test::UnregisteredSet()); },
                       Error::noops),
          "a zonotope minus a custom set");
    check(test::raises([] { test::UnregisteredSet().minkowski_difference(Interval::origin(2)); },
                       Error::noops),
          "a custom minuend");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(I).minkowski_difference(I).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_difference_of_any_pair);
    errors_and_their_switch();
    return test::finish("cont_set minkowski_difference");
}
