// test_cont_set_uminus - ContSet::uminus: the negated set, in the class of the set
//
// CORA: contSet/@contSet/uminus.m

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

namespace {

/// -S has the center -c and the generators -G for a zonotope, the bounds [-sup, -inf] for a box.
void uminus_negates(const std::string &b) {
    const Zonotope Z(column({1.0, 0.0}), Tensor::from_data({0.5, 0.0, 0.2, 0.5}, {2, 2}));
    const Interval I(column({-1.0, 0.0}), column({2.0, 3.0}));
    const Set A = static_cast<const ContSet &>(Z).uminus(),
              B = static_cast<const ContSet &>(I).uminus();
    check(A.is<Zonotope>() && test::close(A.as<Zonotope>().center(), {-1.0, 0.0}) &&
              test::close(A.as<Zonotope>().generators(), {-0.5, -0.0, -0.2, -0.5}),
          b + ": a zonotope");
    check(B.is<Interval>() && test::close(B.as<Interval>().inf(), {-2.0, -3.0}) &&
              test::close(B.as<Interval>().sup(), {1.0, 0.0}),
          b + ": a box");
    for (const std::vector<double> &d : {std::vector<double>{1.0, 2.0}, {-1.0, 0.5}}) {
        const std::vector<double> minus_d{-d[0], -d[1]};
        check(test::close(test::support(A, d), test::support(Z, minus_d)),
              b + ": the support of -Z");
    }
    // the unbounded box has no NaN from a zero entry of the map
    const double inf = std::numeric_limits<double>::infinity();
    const Interval U(column({-inf, 0.0}), column({1.0, inf}));
    const Interval negated = static_cast<const ContSet &>(U).uminus().as<Interval>();
    check(negated.inf().data() == std::vector<double>({-1.0, -inf}) &&
              negated.sup().data() == std::vector<double>({inf, 0.0}),
          b + ": an unbounded box");
}

/// The empty set and the whole space are their own negation.
void uminus_of_the_extremes() {
    const Set E = static_cast<const ContSet &>(EmptySet(3)).uminus();
    const Set F = static_cast<const ContSet &>(Fullspace(3)).uminus();
    check(E.is<EmptySet>() && E->dim() == 3, "the empty set");
    check(F.is<Fullspace>() && F->dim() == 3, "the whole space");
}

/// A class without the linear map raises noops; the checks off change nothing.
void errors_and_their_switch() {
    check(test::raises([] { test::UnregisteredSet().uminus(); }, Error::noops), "noops");
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    ChecksScope off(false);
    check(static_cast<const ContSet &>(I).uminus().is<Interval>(), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(uminus_negates);
    uminus_of_the_extremes();
    errors_and_their_switch();
    return test::finish("cont_set uminus");
}
