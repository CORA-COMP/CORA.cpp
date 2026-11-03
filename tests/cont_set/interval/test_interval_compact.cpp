// test_interval_compact - Interval::compact_impl: a box is returned as it is
//
// CORA: contSet/@contSet/compact.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "testing.h"

using namespace cora;
using test::check;

int main() {
    const Interval S(test::column({0.0, -1.0}), test::column({1.0, 2.0}));
    // CORA returns before it reads the method and tol: any of them is accepted
    const Set R = static_cast<const ContSet &>(S).compact("anything", -1.0);
    check(R.is<Interval>() && R->dim() == S.dim(), "Interval: the same class and dimension");
    check(test::close(R.as<Interval>().inf(), S.inf(), 0) &&
              test::close(R.as<Interval>().sup(), S.sup(), 0),
          "Interval: the same bounds");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(S).compact().is<Interval>(), "the checks off");
    return test::finish("interval compact");
}
