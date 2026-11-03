// test_cont_set_lift - ContSet::lift: the checked entry of every set's lift_
//
// CORA: contSet/@contSet/lift.m
// CORA: contSet/@contSet/lift_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"
#include "unregistered_set.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

const double kInf = std::numeric_limits<double>::infinity();

/// A box keeps its coordinates at proj (the identity by default) and is unbounded elsewhere; a
/// zonotope is permuted (and needs N = n).
void lift_by_class(const std::string &b) {
    const Interval I(column({0.0, 2.0}), column({1.0, 3.0}));
    const Set A = static_cast<const ContSet &>(I).lift(4, std::vector<int64_t>{3, 1});
    check(A.is<Interval>() &&
              A.as<Interval>().inf().data() == std::vector<double>{-kInf, 2.0, -kInf, 0.0} &&
              A.as<Interval>().sup().data() == std::vector<double>{kInf, 3.0, kInf, 1.0},
          b + ": a box at positions 3 and 1");
    const Set D = static_cast<const ContSet &>(I).lift(3);
    check(D.is<Interval>() && D.as<Interval>().inf().data() == std::vector<double>{0.0, 2.0, -kInf},
          b + ": the default positions 0..n-1");

    const Zonotope Z(column({1.0, 2.0, 3.0}),
                     Tensor::from_data({1.0, 0.0, 0.0, 2.0, 0.0, 0.0}, {3, 2}));
    const Set P = static_cast<const ContSet &>(Z).lift(3, std::vector<int64_t>{1, 2, 0});
    check(P.is<Zonotope>() && test::close(P.as<Zonotope>().center(), {3.0, 1.0, 2.0}),
          b + ": a zonotope is permuted");
    check(static_cast<const ContSet &>(Z).lift(3).as<Zonotope>().isequal(Z),
          b + ": the default positions of a zonotope");
    const Zonotope batch = Zonotope::stack({Z, Z + column({1.0, 1.0, 1.0})});
    check(static_cast<const ContSet &>(batch)
                  .lift(3, std::vector<int64_t>{1, 2, 0})
                  .as<Zonotope>()
                  .center()
                  .shape() == Shape({2, 3, 1}),
          b + ": a batch of zonotopes");
}

/// The empty set and the whole space take the new dimension.
void lift_the_extremes() {
    const Set E = static_cast<const ContSet &>(EmptySet(2)).lift(4, std::vector<int64_t>{1, 3});
    const Set F = static_cast<const ContSet &>(Fullspace(2)).lift(4);
    check(E.is<EmptySet>() && E->dim() == 4, "the empty set");
    check(F.is<Fullspace>() && F->dim() == 4, "the whole space");
}

/// CORA's checks (the same wrong_value for all), the class's own failure, noops, checks off.
void errors_and_their_switch() {
    const Interval I(column({0.0, 2.0}), column({1.0, 3.0}));
    const ContSet &S = I;
    check(test::raises([&] { S.lift(1); }, Error::wrong_value), "N below the dimension");
    check(test::raises([&] { S.lift(3, std::vector<int64_t>{0}); }, Error::wrong_value),
          "one position short");
    check(test::raises([&] { S.lift(3, std::vector<int64_t>{0, 3}); }, Error::wrong_value),
          "a position beyond N");
    check(test::raises([&] { S.lift(3, std::vector<int64_t>{-1, 0}); }, Error::wrong_value),
          "a negative position");
    check(test::raises([] { test::UnregisteredSet().lift(3); }, Error::noops), "noops");
    const Zonotope Z(column({1.0, 2.0}), Tensor::from_data({1.0, 0.0}, {2, 1}));
    check(test::raises([&] { static_cast<const ContSet &>(Z).lift(3); }, Error::not_defined),
          "a zonotope cannot have unbounded dimensions");
    check(test::raises([&] { static_cast<const ContSet &>(Z).lift(2, std::vector<int64_t>{1, 1}); },
                       Error::wrong_value),
          "a position twice");
    ChecksScope off(false);
    check(S.lift(3).is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(lift_by_class);
    lift_the_extremes();
    errors_and_their_switch();
    return test::finish("cont_set lift");
}
