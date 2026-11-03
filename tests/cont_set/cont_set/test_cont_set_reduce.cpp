// test_cont_set_reduce - ContSet::reduce: an enclosure with fewer generators, in the class of the set
//
// CORA: contSet/@contSet/reduce.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// A zonotope of seven generators reduces to two (order 1 in two dimensions) and still contains
/// itself; a box stays the same box.
void reduce_keeps_the_class_and_encloses(const std::string &b) {
    Rng rng(5);
    const Zonotope Z = Zonotope::generate_random(2, 7, rng);
    const ContSet &S = Z;
    const Set R = S.reduce(1);
    check(R.is<Zonotope>() && R.as<Zonotope>().generators().shape().back() == 2,
          b + ": two generators remain");
    for (int k = 0; k < 6; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        check(test::support(R, d) >= test::support(Z, d) - 1e-12, b + ": the reduction encloses");
    }
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0}));
    const Set B = static_cast<const ContSet &>(I).reduce(3);
    check(B.is<Interval>() && test::close(B.as<Interval>().sup(), I.sup(), 0.0),
          b + ": a box is its own reduction");
}

/// An order below 1 is refused; a class without reduce raises noops; the checks off skip the order.
void errors_and_their_switch() {
    const Interval I(column({0.0}), column({1.0}));
    const ContSet &S = I;
    check(test::raises([&] { S.reduce(0); }, Error::wrong_value), "order 0");
    check(test::raises([] { test::UnregisteredSet().reduce(1); }, Error::noops), "noops");
    ChecksScope off(false);
    check(S.reduce(0).is<Interval>(), "the checks off skip the order");
}

} // namespace

int main() {
    test::for_each_backend(reduce_keeps_the_class_and_encloses);
    errors_and_their_switch();
    return test::finish("cont_set reduce");
}
